#!/usr/bin/env bash
set -euo pipefail
exec uvicorn ki_geodaten.app.main:app \
    --host 127.0.0.1 --port 8000 \
    --reload \
    --reload-dir ki_geodaten \
    --reload-exclude 'data/*' \
    --reload-exclude '*.db*'
