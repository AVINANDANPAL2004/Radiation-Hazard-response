#!/usr/bin/env bash
# One command: builds the React app when it is missing or older than its sources, then serves everything.
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
stale=""
if [ -f frontend/dist/index.html ]; then
  stale="$(find frontend/src frontend/index.html frontend/package.json -newer frontend/dist/index.html -print -quit)"
fi
if [ ! -f frontend/dist/index.html ] || [ -n "$stale" ]; then
  echo "Building frontend..."
  (cd frontend && npm ci --no-audit --no-fund && npm run build)
fi
echo "Open http://localhost:${PORT}"
exec python3 -m uvicorn server:app --host 127.0.0.1 --port "$PORT"
