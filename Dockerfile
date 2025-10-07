FROM python:3.12-alpine AS deps
WORKDIR /deps
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.12-alpine
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apk add --no-cache ffmpeg su-exec \
    && addgroup -S app \
    && adduser -S app -G app \
    && mkdir -p /app/jobs /app/Downloads
COPY --from=deps /opt/venv /opt/venv
COPY app.py ./
COPY requirements.txt ./
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
