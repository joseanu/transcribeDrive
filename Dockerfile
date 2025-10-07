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
RUN apk add --no-cache ffmpeg \
    && addgroup -S app \
    && adduser -S app -G app
COPY --from=deps /opt/venv /opt/venv
COPY app.py ./
COPY requirements.txt ./
USER app
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
