# Dashboard (React + Vite)

Talks to the FastAPI backend in the parent folder (`server.py`).

    npm install
    npm run dev      # http://localhost:5173, /api is proxied to http://127.0.0.1:8000
    npm run build    # writes dist/, which the backend serves at http://localhost:8000

Set `VITE_API_BASE` (e.g. `http://myhost:8000`) to point a separately hosted build at the API.
