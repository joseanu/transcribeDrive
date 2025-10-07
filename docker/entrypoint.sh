#!/bin/sh
set -e

: "${JOBS_DIR:=/app/jobs}"
: "${DOWNLOAD_VIDEO_DIR:=/app/Downloads}"

mkdir -p "$JOBS_DIR" "$DOWNLOAD_VIDEO_DIR"
chown -R app:app "$JOBS_DIR" "$DOWNLOAD_VIDEO_DIR"

exec su-exec app "$@"
