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

## v2 training and dashboard (algorithm + map refinements)

What changed, and why:

- **Timeouts are no longer terminal.** An episode cut off at 50 steps is a truncation. The old code set `done` on the last transition regardless, which told the agent that running out of time was as final as reaching the goal. Only `success` and `overdose` terminate now.
- **Best-checkpoint selection on validation seeds.** Every 100 episodes the agent is scored on 50 greedy validation scenarios (`CKPT_SEED0`, disjoint from the 200 test scenarios). The weights from the best checkpoint are kept and reported. Test seeds are never used to choose a model. The advisory-guided agent swings widely across seeds, so this matters most there.
- The network architecture is unchanged, so v1 weights still load.

Run a v2 experiment without touching v1:

    python run_all.py --tag v2               # writes results_v2/, models_v2/, logs_v2/ (about 2 h on 2 CPU cores)
    python analyze.py v2                     # writes results_table_v2.md, results_v2.csv, figures_v2/
    RAD_RESULTS_DIR=results_v2 RAD_MODELS_DIR=models_v2 python -m uvicorn server:app --port 8000

PowerShell: `$env:RAD_RESULTS_DIR="results_v2"; $env:RAD_MODELS_DIR="models_v2"; python -m uvicorn server:app --port 8000`

Compare the two runs in the Results tab: the "Selected checkpoint" row shows which episode was kept for each agent. Only adopt v2 if its test success beats v1 by more than the seed-to-seed spread (the ± values), not because the code looks better.

Map: the dashboard now shows the sensing window (the 5×5 area the sensors read), cells the safety shield blocked for the current agent (✕), the drift direction of each source (arrow), a fading trail, and a dose-rate colour key. Toggle overlays with the chips above the map.

## Tests

    pip install -r requirements-dev.txt
    pytest                # from radiation_agent/

The environment tests check simulator invariants and fingerprint the layouts behind the reported evaluation (seeds 500000–500199), so any change that alters the simulation fails loudly. The API tests need torch and are skipped when it is not installed. CI runs both backend tests and the frontend build (`.github/workflows/ci.yml`).
