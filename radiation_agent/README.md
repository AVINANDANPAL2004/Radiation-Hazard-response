# Radiation-Hazard Zone Autonomous Agent (Task 5): backend + dashboard

**Simulation only.** Agents: Baseline DRL, CNN-DRL, CNN-LSTM-DRL, LLM-guided CNN-LSTM-DRL (Double DQN).
The "LLM" is a rule-based advisory parser, not a real language model. See `report.md`.

## Quick start
    pip install -r requirements.txt
    ./run.sh                    # builds the frontend if needed, serves everything on http://localhost:8000

## Development (hot reload)
    uvicorn server:app --port 8000        # terminal 1: backend (live model inference)
    cd frontend && npm install && npm run dev   # terminal 2: http://localhost:5173

## Layout
| Path | Purpose |
|---|---|
| `server.py` | FastAPI: `/api/compare`, `/api/episode`, `/api/summary`, `/api/curves`, `/api/health`; serves `frontend/dist` |
| `env.py`, `advisory.py`, `models.py` | simulation + safety shield, advisory parser, Q-networks |
| `train.py`, `run_all.py`, `analyze.py` | training (4 agents x 3 seeds), evaluation, tables and figures |
| `models/`, `results/` | 12 trained models, per-run learning curves and final metrics |
| `frontend/` | React + Vite dashboard |
| `report.md`, `figures/` | technical report and plots |

## API
- `GET /api/compare?seed=500003&model_seed=0`: runs all four agents live on one scenario (any non-negative seed gives a new layout; 500000-500199 is the evaluation set).
- `GET /api/episode?agent=cnn&seed=...&model_seed=...`: one agent.
- `GET /api/summary`, `GET /api/curves`: results from `results/`.

Retraining: `python run_all.py` (about 2 hours on 2 CPU cores), then `python analyze.py`.
