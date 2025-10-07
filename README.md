# transcribeDrive

FastAPI service that accepts Google Drive video URLs, extracts MP3 audio with `ffmpeg`, uploads it to AssemblyAI, polls for results, and delivers the transcript to a caller-provided callback URL.

## Configuration

Set these environment variables before starting the service:

- `SERVICE_AUTH_TOKEN` – bearer token required to call `POST /transcribe`.
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN` – OAuth credentials with Google Drive read access.
- `ASSEMBLYAI_API_KEY` – AssemblyAI API key.
- Optional tuning:
  - `MAX_VIDEO_MB` – reject downloads larger than this size.
  - `LOG_LEVEL` – set to `DEBUG` for extra diagnostics.
  - `JOBS_DIR` – working directory for job files (default `./jobs`).
  - `KEEP_FAILED_JOBS` – keep failed job folders on disk for inspection (default disabled).
  - `DOWNLOAD_VIDEO_FIRST` – set to `true` to save the Drive video locally before extracting audio.
  - `DOWNLOAD_VIDEO_DIR` – where to save the video when the previous flag is enabled (default `/app/Downloads`).
  - `POLL_INTERVAL_SEC` – seconds between AssemblyAI status checks (default 10).
  - `POLL_TIMEOUT_MIN` – stop polling after this many minutes (default 120).
  - `REQUEST_TIMEOUT_SEC` – per-request timeout for most HTTP calls (default 30).
  - `DRIVE_DOWNLOAD_TIMEOUT_SEC` – read timeout for the streaming Drive download (default 300).

## Running locally

1. Install dependencies: `pip install -r requirements.txt`.
2. Ensure `ffmpeg` is available on your `PATH`.
3. Start the app:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8080
   ```

4. Health check: `GET /healthz` returns `{ "status": "ok" }`.

## Request example

```bash
curl -X POST http://localhost:8080/transcribe \
  -H "Authorization: Bearer $SERVICE_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "video_url": "https://drive.google.com/file/d/14EjpRa8v01eHNbZx55PcFLg_z0c6huQj/view",
        "callback_url": "https://example.com/webhooks/transcripts",
        "language_code": "es",
        "speaker_labels": true,
        "format_text": true,
        "punctuate": true,
        "speech_model": "universal"
      }'
```

The service replies with `202 Accepted` and a `job_id`, then processes the job in the background and POSTs the result or error to the `callback_url`.

Optional request fields:
- `language_code` (or legacy `language`) – AssemblyAI language code like `es`.
- `speaker_labels` – enable diarization.
- `format_text` and `punctuate` – request formatted, punctuated output.
- `speech_model` – override the default model (e.g. `universal`).

## Docker image

Build and run the container:

```bash
docker build -t transcribe-drive .
docker run --rm -p 8080:8080 \
  -e SERVICE_AUTH_TOKEN=... \
  -e GOOGLE_CLIENT_ID=... \
  -e GOOGLE_CLIENT_SECRET=... \
  -e GOOGLE_REFRESH_TOKEN=... \
  -e ASSEMBLYAI_API_KEY=... \
  transcribe-drive
```

The Docker image uses `python:3.12-alpine`, installs only FastAPI, Uvicorn, and Requests, and brings in `ffmpeg` via `apk`. For a smaller image you can swap the `apk add ffmpeg` step for a static binary download and copy only the executable into the runtime layer.


## Docker Compose

1. Copy `.env.example` to `.env` and fill in your secrets.
2. Run `docker compose up --build`.

The service listens on port 8080 by default.


During debugging, the service now preserves `ffmpeg_stderr.log` and, if OAuth fails, `google_token_error.json` under each `jobs/<job_id>` when `KEEP_FAILED_JOBS=true`. Work directories are only deleted after success or when the flag is false.

When `DOWNLOAD_VIDEO_FIRST=true`, the raw Google Drive file is saved in `DOWNLOAD_VIDEO_DIR` (default `/app/Downloads`) before running ffmpeg, so you can inspect or reprocess it manually if streaming conversion continues to fail.

If streaming extraction fails because the Drive file is not stream-friendly, the service automatically falls back to downloading the full video inside `jobs/<job_id>` and converts it from disk.

Auto-downloaded videos now land in `/app/Downloads` by default; the conversions and fallback MP3s stay under `/app/jobs`.
