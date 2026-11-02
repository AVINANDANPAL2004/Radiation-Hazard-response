#!/usr/bin/env bash
# One command: builds the React app (if needed) and starts the backend that serves it.
set -e
cd "$(dirname "$0")"
if [ ! -d frontend/dist ]; then
  (cd frontend && npm install && npm run build)
fi
echo "Open http://localhost:8000"
exec python3 -m uvicorn server:app --port 8000
