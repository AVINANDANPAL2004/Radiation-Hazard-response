# Radiation-Hazard-response
Autonomous Radiation-Hazard Zone Agent | AI-driven risk analysis, hazard simulation, and intelligent decision-making.
Radiation-Hazard Zone Autonomous Agent
A simulation-based research project that compares four deep-reinforcement-learning agents navigating a radiation-hazard grid. It includes a FastAPI backend, trained PyTorch models, evaluation outputs, and a React/Vite dashboard for comparing agents and replaying episodes.
Safety notice: This is a research simulation only. It is not validated for nuclear facilities, emergency response, or real-world robotic operation. The advisory component is a deterministic rule-based parser, not a live LLM. Do not use these results to make real safety decisions.

Dashboard and API
Once deployed, open the service's Render URL to use the dashboard. The backend exposes:
Endpoint	Description
GET /api/health	Service status, available agents, model seeds, action names, and evaluation seed range
GET /api/compare?seed=500003&model_seed=0	Runs all four agents on the same seeded scenario
GET /api/episode?agent=cnn&seed=500003&model_seed=0	Runs one selected agent on a scenario
GET /api/summary	Final evaluation metrics aggregated across saved runs
GET /api/curves	Training and evaluation learning curves


Available agent identifiers: baseline, cnn, cnn_lstm, and llm_cnn_lstm. Available model seeds are listed by /api/health.
Agents compared
- Baseline DRL: Double DQN using the environment's sensor vector.
- CNN-DRL: Adds a convolutional visual perception branch.
- CNN-LSTM-DRL: Adds temporal memory over recent observations.
- Advisory-guided CNN-LSTM-DRL: Adds bounded features parsed from simulated natural-language hazard advisories. The current implementation uses a rule-based parser rather than a real language model.
The simulation uses a 10×10 grid, walls, drifting radiation sources, a cumulative exposure budget, and a safety mask that blocks moves into walls or cells whose immediate dose exceeds the configured threshold. All agents are evaluated on the same seeded scenarios.
Reported evaluation snapshot
The included report records results over 200 held-out scenarios per run and three training seeds. The baseline agent achieved 78.00% ± 1.78% success, CNN-DRL 56.00% ± 1.87%, CNN-LSTM-DRL 9.83% ± 4.78%, and advisory-guided CNN-LSTM-DRL 26.17% ± 11.49%. These are results from the supplied experiment artifacts, not a guarantee of generalization. The recurrent agents were likely undertrained within the experiment budget; consult [`report.md`](report.md) for methodology, caveats, and analysis.
Run locally
Requirements
- Python 3.11 recommended
- Node.js 20 or later and npm
- CPU is sufficient for inference; training is much more computationally expensive
Start the full app
From the project root:
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
python -m uvicorn server:app --host 127.0.0.1 --port 8000
Open http://127.0.0.1:8000.

For frontend development with hot reload, run the backend in one terminal and then run cd frontend && npm run dev in another. Vite proxies /api requests to http://127.0.0.1:8000.
Deploy to Render
The repository includes [`render.yaml`](render.yaml) for a Render Python web service. Connect the GitHub repository to Render and create a service from the Blueprint, or create a Python web service with:
- Build command: pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements.txt && cd frontend && npm ci && npm run build
- Start command: python -m uvicorn server:app --host 0.0.0.0 --port $PORT
- Health check path: /api/health
- Region: Singapore is a reasonable choice for users in India.
The first build can take several minutes because PyTorch and the frontend dependencies must be installed. Live inference is CPU-based. The free instance may have cold starts or resource limits; if the build or runtime runs out of memory, use a larger instance or separate the frontend and API.
Project structure
.
├── server.py              # FastAPI endpoints and production static-file serving
├── env.py                 # Radiation-zone simulation and safety shield
├── advisory.py            # Rule-based advisory parser (not a real LLM)
├── models.py              # PyTorch Q-networks
├── train.py               # Double-DQN training/evaluation utilities
├── run_all.py             # Reproduce the multi-agent, multi-seed experiments
├── analyze.py              # Aggregate results and regenerate plots/tables
├── models/                 # Saved trained model weights
├── results/                # Per-seed evaluation and learning-curve JSON
├── logs/                   # Experiment logs
├── figures/                # Generated plots
├── frontend/               # React + Vite dashboard
├── report.md               # Methods, results, limitations, and interpretation
└── results_table.md        # Compact metrics table
Reproduce the experiments
Training is not needed to run the dashboard because model weights and result files are included. To retrain all agents and regenerate the analysis:
pip install -r requirements.txt
python run_all.py
python analyze.py
Training can take hours on a CPU. Run experiments in a suitable compute environment and keep the existing model/result files backed up if you want to preserve the included snapshot.
Limitations
- All results are simulation-based and depend on the included environment and experiment settings.
- The advisory parser is deterministic and rule-based; no LLM inference was performed for the reported results.
- The safety mask is a simulator rule, not a validated real-world safety system.
- The LSTM variants show substantially lower success in this training budget and should not be presented as superior agents.
- Real deployment would require independent validation, appropriate sensors, safety engineering, and authorized oversight.
License
No license has been added. Unless a license is added, standard copyright applies and others do not automatically receive permission to reuse, modify, or redistribute the code. Add a license only if you have the rights and want to grant those permissions.
