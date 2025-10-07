import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, HttpUrl


logger = logging.getLogger("transcribeDrive")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


class TranscribeRequest(BaseModel):
    video_url: HttpUrl
    callback_url: HttpUrl
    language: Optional[str] = None
    language_code: Optional[str] = None
    speaker_labels: Optional[bool] = False
    format_text: Optional[bool] = None
    punctuate: Optional[bool] = None
    speech_model: Optional[str] = None


@dataclass
class Settings:
    service_auth_token: str
    google_client_id: str
    google_client_secret: str
    google_refresh_token: str
    assemblyai_api_key: str
    jobs_dir: str
    keep_failed_jobs: bool = False
    download_video_first: bool = False
    download_video_dir: str = ""
    poll_interval_sec: int = 10
    poll_timeout_sec: int = 120 * 60
    request_timeout_sec: int = 30
    drive_download_timeout_sec: int = 300
    max_video_mb: Optional[float] = None

    @classmethod
    def from_env(cls) -> "Settings":
        def require(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise RuntimeError(f"Missing required environment variable: {name}")
            return value

        poll_interval = int(os.getenv("POLL_INTERVAL_SEC", "10"))
        poll_timeout = int(float(os.getenv("POLL_TIMEOUT_MIN", "120")) * 60)
        request_timeout = int(os.getenv("REQUEST_TIMEOUT_SEC", "30"))
        drive_timeout = int(os.getenv("DRIVE_DOWNLOAD_TIMEOUT_SEC", "300"))
        max_video_env = os.getenv("MAX_VIDEO_MB")
        max_video_mb = float(max_video_env) if max_video_env else None
        jobs_dir = os.getenv("JOBS_DIR") or os.path.join(os.getcwd(), "jobs")
        jobs_dir = os.path.abspath(jobs_dir)
        keep_failed_jobs = os.getenv("KEEP_FAILED_JOBS", "false").lower() in {"1", "true", "yes"}
        download_video_first = os.getenv("DOWNLOAD_VIDEO_FIRST", "false").lower() in {"1", "true", "yes"}
        download_video_dir = os.getenv("DOWNLOAD_VIDEO_DIR") or os.path.join(os.path.expanduser("~"), "Downloads")
        download_video_dir = os.path.abspath(download_video_dir)

        return cls(
            service_auth_token=require("SERVICE_AUTH_TOKEN"),
            google_client_id=require("GOOGLE_CLIENT_ID"),
            google_client_secret=require("GOOGLE_CLIENT_SECRET"),
            google_refresh_token=require("GOOGLE_REFRESH_TOKEN"),
            assemblyai_api_key=require("ASSEMBLYAI_API_KEY"),
            jobs_dir=jobs_dir,
            keep_failed_jobs=keep_failed_jobs,
            download_video_first=download_video_first,
            download_video_dir=download_video_dir,
            poll_interval_sec=poll_interval,
            poll_timeout_sec=poll_timeout,
            request_timeout_sec=request_timeout,
            drive_download_timeout_sec=drive_timeout,
            max_video_mb=max_video_mb,
        )


settings = Settings.from_env()
os.makedirs(settings.jobs_dir, exist_ok=True)
if settings.download_video_first:
    os.makedirs(settings.download_video_dir, exist_ok=True)
app = FastAPI()


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe", status_code=status.HTTP_202_ACCEPTED)
async def transcribe(
    payload: TranscribeRequest,
    background_tasks: BackgroundTasks,
    authorization: str = Header(..., alias="Authorization"),
) -> Dict[str, str]:
    token = _validate_bearer_token(authorization)
    if token != settings.service_auth_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})

    job_id = uuid.uuid4().hex
    background_tasks.add_task(_process_job, job_id, payload.dict())
    logger.info("Accepted transcription job %s", job_id)
    return {"job_id": job_id}


_drive_patterns = (
    re.compile(r"drive\.google\.com/file/d/([^/]+)"),
    re.compile(r"drive\.google\.com/open\?id=([^&]+)"),
    re.compile(r"drive\.google\.com/uc\?id=([^&]+)"),
)


def _validate_bearer_token(header_value: str) -> str:
    if not header_value or not header_value.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})
    return header_value.split(" ", 1)[1]


def _process_job(job_id: str, payload: Dict[str, object]) -> None:
    job_dir = os.path.join(settings.jobs_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    logger.info("Job %s started; work dir %s", job_id, job_dir)
    success = False
    try:
        callback_url = str(payload["callback_url"])
        language = payload.get("language_code") or payload.get("language") or None
        speaker_labels = bool(payload.get("speaker_labels")) if payload.get("speaker_labels") is not None else False
        format_text = payload.get("format_text") if payload.get("format_text") is not None else None
        punctuate = payload.get("punctuate") if payload.get("punctuate") is not None else None
        speech_model = payload.get("speech_model") or None

        logger.info("Job %s downloading source video", job_id)
        audio_path = _download_and_convert(job_id, str(payload["video_url"]), job_dir)
        logger.info("Job %s extracted audio to %s", job_id, audio_path)
        logger.info("Job %s uploading audio to AssemblyAI", job_id)
        upload_url = _upload_audio(audio_path)
        logger.info("Job %s audio uploaded", job_id)
        transcript_data = _transcribe_audio(job_id, upload_url, language, speaker_labels, format_text, punctuate, speech_model)

        status_value = transcript_data.get("status")
        transcript_id = transcript_data.get("id")
        if status_value == "completed":
            callback_payload = {
                "job_id": job_id,
                "transcript_id": transcript_id,
                "status": "completed",
                "text": transcript_data.get("text", ""),
                "duration_seconds": transcript_data.get("audio_duration", 0),
                "words": transcript_data.get("words", []),
            }
            success = True
        else:
            callback_payload = {
                "job_id": job_id,
                "transcript_id": transcript_id,
                "status": "error",
                "error": transcript_data.get("error", "Transcription failed"),
            }

        _send_callback(callback_url, callback_payload)
        logger.info("Job %s finished with status %s", job_id, callback_payload["status"])
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Job %s failed: %s", job_id, exc)
        callback_url = str(payload.get("callback_url"))
        error_payload = {
            "job_id": job_id,
            "status": "error",
            "error": str(exc),
        }
        if "transcript_id" in locals():
            error_payload["transcript_id"] = locals()["transcript_id"]
        _send_callback(callback_url, error_payload)
    finally:
        if success or not settings.keep_failed_jobs:
            _cleanup(job_dir)
        else:
            logger.info("Job %s failed; preserving work dir %s for inspection", job_id, job_dir)

def _download_and_convert(job_id: str, video_url: str, work_dir: str) -> str:
    file_id = _extract_drive_file_id(video_url)
    os.makedirs(work_dir, exist_ok=True)
    audio_path = os.path.join(work_dir, "audio.mp3")

    min_audio_bytes = 16 * 1024

    def _convert_local_file(video_path: str, source_label: str) -> None:
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "libmp3lame",
            "-b:a",
            "64k",
            "-f",
            "mp3",
            audio_path,
        ]
        logger.debug("Job %s converting %s with ffmpeg: %s", job_id, source_label, " ".join(ffmpeg_cmd))
        start_time = time.monotonic()
        proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        elapsed = time.monotonic() - start_time
        if proc.returncode != 0:
            logger.error("Job %s ffmpeg stderr: %s", job_id, proc.stderr.strip())
            raise RuntimeError(f"ffmpeg conversion failed with code {proc.returncode}")
        if proc.stderr.strip():
            ffmpeg_log = os.path.join(work_dir, "ffmpeg_stderr.log")
            with open(ffmpeg_log, "w", encoding="utf-8") as handle:
                handle.write(proc.stderr)
            logger.debug("Job %s ffmpeg warnings captured in %s", job_id, ffmpeg_log)
        audio_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        logger.info("Job %s encoded audio (%.2f MB) from %s in %.1fs", job_id, audio_size / (1024 * 1024), source_label, elapsed)
        if audio_size < min_audio_bytes:
            raise RuntimeError("Audio file is too small; source may lack audio track")

    def _download_video_file(target_dir: str, source: str) -> str:
        os.makedirs(target_dir, exist_ok=True)
        video_path = os.path.join(target_dir, f"{job_id}.mp4")
        try:
            os.remove(video_path)
        except FileNotFoundError:
            pass

        downloaded_bytes = 0
        for attempt in range(2):
            access_token = _fetch_google_access_token(job_id, work_dir)
            download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
            headers = {"Authorization": f"Bearer {access_token}"}
            timeout = (settings.request_timeout_sec, settings.drive_download_timeout_sec)
            with requests.get(download_url, headers=headers, stream=True, timeout=timeout) as response:  # type: ignore[arg-type]
                if response.status_code in (401, 403):
                    logger.warning("Job %s received %s from Drive while downloading %s; retrying", job_id, response.status_code, source)
                    continue
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if settings.max_video_mb and content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > settings.max_video_mb:
                        raise RuntimeError(f"Video size {size_mb:.2f} MB exceeds limit of {settings.max_video_mb} MB")
                downloaded_bytes = 0
                with open(video_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded_bytes += len(chunk)
                        if settings.max_video_mb and downloaded_bytes > settings.max_video_mb * 1024 * 1024:
                            raise RuntimeError("Downloaded data exceeds configured size limit")
            if downloaded_bytes == 0:
                logger.warning("Job %s downloaded 0 bytes for %s; retrying", job_id, source)
                continue
            logger.info("Job %s saved %s to %s (%.2f MB)", job_id, source, video_path, downloaded_bytes / (1024 * 1024))
            return video_path
        raise RuntimeError("Unauthorized to download file from Google Drive")

    def _stream_to_ffmpeg() -> None:
        for attempt in range(2):
            access_token = _fetch_google_access_token(job_id, work_dir)
            download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
            headers = {"Authorization": f"Bearer {access_token}"}
            timeout = (settings.request_timeout_sec, settings.drive_download_timeout_sec)
            with requests.get(download_url, headers=headers, stream=True, timeout=timeout) as response:  # type: ignore[arg-type]
                if response.status_code in (401, 403):
                    logger.warning("Job %s received %s from Drive; will try refreshing token", job_id, response.status_code)
                    continue
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if settings.max_video_mb and content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > settings.max_video_mb:
                        raise RuntimeError(f"Video size {size_mb:.2f} MB exceeds limit of {settings.max_video_mb} MB")

                ffmpeg_cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    "pipe:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-acodec",
                    "libmp3lame",
                    "-b:a",
                    "64k",
                    "-f",
                    "mp3",
                    audio_path,
                ]
                logger.debug("Job %s launching ffmpeg: %s", job_id, " ".join(ffmpeg_cmd))
                start_time = time.monotonic()
                proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                assert proc.stdin is not None  # for type checkers

                written_bytes = 0
                try:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        proc.stdin.write(chunk)
                        written_bytes += len(chunk)
                        if settings.max_video_mb and written_bytes > settings.max_video_mb * 1024 * 1024:
                            raise RuntimeError("Downloaded data exceeds configured size limit")
                finally:
                    proc.stdin.close()

                if written_bytes == 0:
                    raise RuntimeError("No data received from Google Drive stream")

                return_code = proc.wait()
                elapsed = time.monotonic() - start_time
                stderr_output = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
                if return_code != 0:
                    logger.error("Job %s ffmpeg stderr: %s", job_id, stderr_output.strip())
                    raise RuntimeError(f"ffmpeg conversion failed with code {return_code}")
                if stderr_output.strip():
                    ffmpeg_log = os.path.join(work_dir, "ffmpeg_stderr.log")
                    with open(ffmpeg_log, "w", encoding="utf-8") as handle:
                        handle.write(stderr_output)
                    logger.debug("Job %s ffmpeg warnings captured in %s", job_id, ffmpeg_log)
                logger.info("Job %s downloaded %.2f MB and encoded audio in %.1fs", job_id, written_bytes / (1024 * 1024), elapsed)
                return
        raise RuntimeError("Unauthorized to download file from Google Drive")

    def _validate_audio_size() -> None:
        if not os.path.exists(audio_path):
            raise RuntimeError("Audio file was not created")
        size_bytes = os.path.getsize(audio_path)
        if size_bytes < min_audio_bytes:
            raise RuntimeError("Audio file is too small; source may lack audio track")

    if settings.download_video_first:
        video_path = _download_video_file(settings.download_video_dir, "video")
        _convert_local_file(video_path, "downloaded video")
        return audio_path

    try:
        _stream_to_ffmpeg()
        _validate_audio_size()
        return audio_path
    except RuntimeError as exc:
        if "Audio file is too small" not in str(exc) and "No data received" not in str(exc):
            raise
        logger.warning("Job %s streaming conversion failed (%s); retrying with full download", job_id, exc)
        video_path = _download_video_file(work_dir, "fallback video")
        _convert_local_file(video_path, "fallback video")
        return audio_path



def _extract_drive_file_id(url: str) -> str:
    for pattern in _drive_patterns:
        match = pattern.search(url)
        if match:
            return match.group(1)
    raise RuntimeError("Unsupported Google Drive URL format")


def _fetch_google_access_token(job_id: str, work_dir: str) -> str:
    data = {
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "refresh_token": settings.google_refresh_token,
        "grant_type": "refresh_token",
    }
    response = requests.post("https://oauth2.googleapis.com/token", data=data, timeout=settings.request_timeout_sec)
    if not response.ok:
        logger.error("Job %s token exchange failed (%s): %s", job_id, response.status_code, response.text[:500])
        error_path = os.path.join(work_dir, "google_token_error.json")
        try:
            payload = response.json()
        except ValueError:
            payload = {"raw": response.text}
        with open(error_path, "w", encoding="utf-8") as handle:
            json.dump({"status": response.status_code, "payload": payload}, handle, indent=2)
        response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("Google OAuth token response missing access_token")
    logger.debug("Obtained Google access token expiring in %ss", payload.get('expires_in'))
    return token


def _upload_audio(audio_path: str) -> str:
    file_size = os.path.getsize(audio_path)
    if file_size <= 0:
        raise RuntimeError("Audio file is empty; ffmpeg may have failed")
    logger.info("Preparing to upload audio (%s, %.2f MB)", audio_path, file_size / (1024 * 1024))

    headers = {
        "authorization": settings.assemblyai_api_key,
        "content-type": "application/octet-stream",
    }
    with open(audio_path, "rb") as handle:
        response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            data=handle,
            timeout=(settings.request_timeout_sec, settings.drive_download_timeout_sec),
        )
    if not response.ok:
        logger.error("AssemblyAI upload failed (%s): %s", response.status_code, response.text[:500])
        response.raise_for_status()
    payload = response.json()
    upload_url = payload.get("upload_url")
    if not upload_url:
        raise RuntimeError("Upload response missing upload_url")
    logger.info("AssemblyAI upload accepted (url tail: %s)", upload_url[-32:])
    return upload_url


def _transcribe_audio(
    job_id: str,
    upload_url: str,
    language: Optional[str],
    speaker_labels: bool,
    format_text: Optional[bool],
    punctuate: Optional[bool],
    speech_model: Optional[str],
) -> Dict[str, object]:
    headers = {
        "authorization": settings.assemblyai_api_key,
        "content-type": "application/json",
    }
    body: Dict[str, object] = {
        "audio_url": upload_url,
    }
    if language:
        body["language_code"] = language
    if speaker_labels:
        body["speaker_labels"] = True
    if format_text is not None:
        body["format_text"] = format_text
    if punctuate is not None:
        body["punctuate"] = punctuate
    if speech_model:
        body["speech_model"] = speech_model

    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers,
        json=body,
        timeout=settings.request_timeout_sec,
    )
    if not response.ok:
        logger.error("Job %s failed to create transcript (%s): %s", job_id, response.status_code, response.text[:500])
        response.raise_for_status()
    transcript = response.json()
    transcript_id = transcript.get("id")
    logger.info("Job %s created AssemblyAI transcript %s", job_id, transcript_id)
    if not transcript_id:
        raise RuntimeError("Transcript creation response missing id")

    deadline = time.monotonic() + settings.poll_timeout_sec
    while time.monotonic() < deadline:
        poll_response = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
            headers=headers,
            timeout=settings.request_timeout_sec,
        )
        if not poll_response.ok:
            logger.error("Job %s polling transcript %s failed (%s): %s", job_id, transcript_id, poll_response.status_code, poll_response.text[:500])
            poll_response.raise_for_status()
        data = poll_response.json()
        status_value = data.get("status")
        logger.debug("Job %s transcript %s status %s", job_id, transcript_id, status_value)
        if status_value in {"completed", "error"}:
            logger.info("Job %s transcript %s finished with status %s", job_id, transcript_id, status_value)
            return data
        time.sleep(settings.poll_interval_sec)

    raise RuntimeError("Polling timed out before transcription completed")


def _send_callback(callback_url: str, payload: Dict[str, object]) -> None:
    try:
        response = requests.post(callback_url, json=payload, timeout=settings.request_timeout_sec)
        if not response.ok:
            logger.error("Callback to %s failed (%s): %s", callback_url, response.status_code, response.text[:500])
            response.raise_for_status()
        logger.info("Delivered callback to %s with status %s", callback_url, response.status_code)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to deliver callback to %s: %s", callback_url, exc)


def _cleanup(path: str) -> None:
    try:
        shutil.rmtree(path)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to clean up work directory %s: %s", path, exc)


__all__ = ["app"]
