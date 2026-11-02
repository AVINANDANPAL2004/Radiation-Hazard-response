# Intelligent Autonomous Agent for Radiation-Hazard Zone Operations
**Task 5: CNN + LSTM + LLM + Deep Reinforcement Learning (simulation only)**

> All experiments are simulation-based. The agent has an independent rule-based safety shield, the LLM component cannot override it, and nothing here is validated for real nuclear or emergency operations.

## 1. Problem and simulated environment
A robot must travel from the facility entrance (bottom-left) to an inspection point (top-right) of a 10x10 maze (about 12% wall cells, random per episode) containing **two drifting radiation sources** (random intensity, random walk, 20% move chance per step). Dose per step = 0.04 x dose rate of the occupied cell; the **lifetime budget is 1.0** (exceeding it fails the episode). Episodes last at most 50 steps.

| Input | Content |
|---|---|
| Image (4x10x10) | local radiation readings (radius 2 only, noisy), wall blueprint, agent/goal map, advisory hazard map (LLM agent only) |
| Sensor vector | position, goal, dosimeter, 4 neighbour readings, 4 neighbour wall flags, exposure fraction, time |
| Text advisory | e.g. "moderate radiation reported in the north-east sector, source drifting west", refreshed every 5 steps; 15% of reports name a wrong sector |

**Reward:** -0.02 per step, +0.1 per cell of progress toward the goal, -6 x dose, +10 on reaching the goal, -5 on exceeding the dose budget.
**Actions:** up, down, left, right, wait.
**Safety shield (identical for all agents):** masks moves into walls and into cells whose per-step dose exceeds 0.20. The agent cannot select a masked action, and no learned or LLM-derived signal can change the mask.
**Identical scenarios:** layout, source positions and drift are seeded per scenario, and the drift random stream does not depend on the agent's actions, so every agent faces the same world. Training, periodic-check and final-evaluation scenario seeds are disjoint.

## 2. Architecture and the role of each component
| Component | Role |
|---|---|
| **CNN** | Spatial perception of the facility image (walls, local radiation, agent/goal) |
| **LSTM** | Temporal memory over the last 4 frames, which helps with partial observability and drifting sources |
| **LLM** | Interprets the natural-language advisory into 5 bounded features (4 sector hazard levels + a caution flag), added to the vector and painted as an image channel |
| **DRL** | Double DQN (gamma 0.98, Adam 7e-4, batch 64, target sync every 300 updates, episode replay of 500 episodes, epsilon 1.0 to 0.05 over the first 60% of episodes) on top of the fused features |

Four agents were compared: **Baseline DRL** (MLP on the sensor vector only), **CNN-DRL**, **CNN-LSTM-DRL**, **LLM-guided CNN-LSTM-DRL**. Image-based agents use a gated fusion (the image branch starts switched off and the network learns how much to use it).

**Important honesty note on the "LLM".** No LLM was called. The offline sandbox has no model access, so the advisory interpreter is a deterministic regex parser standing in for the LLM (`advisory.py`). It has a `backend="llm"` hook for plugging in a real model, but the reported numbers come from the rule parser only. The measured LLM latency is therefore the parser's, not a real LLM's, which would be orders of magnitude slower.

## 3. Results
200 held-out scenarios per run, greedy policy, mean ± std over **3 training seeds**, 2000 training episodes for every agent.

| Agent | Success % | Eval reward | Mean dose | Over-limit % | Travel (steps) | Shield interventions/ep | Latency (ms) | Params |
|---|---|---|---|---|---|---|---|---|
| Baseline DRL | **78.00 ± 1.78** | **5.04 ± 0.42** | **0.52 ± 0.03** | **13.00 ± 1.87** | **17.16 ± 0.42** | **3.17 ± 0.46** | **0.13 ± 0.02** | 36,357 |
| CNN-DRL | 56.00 ± 1.87 | 1.64 ± 0.56 | 0.59 ± 0.03 | 21.67 ± 3.17 | 19.61 ± 1.33 | 8.40 ± 1.29 | 0.33 ± 0.02 | 101,174 |
| CNN-LSTM-DRL | 9.83 ± 4.78 | -4.22 ± 0.48 | 0.61 ± 0.01 | 24.83 ± 1.70 | 30.62 ± 2.98 | 27.12 ± 1.28 | 0.99 ± 0.04 | 233,270 |
| LLM-guided CNN-LSTM-DRL | 26.17 ± 11.49 | -2.23 ± 1.16 | 0.64 ± 0.02 | 24.00 ± 3.89 | 24.10 ± 1.40 | 22.94 ± 4.99 | 1.06 ± 0.03 | 233,270 |

Per-seed success: baseline 0.805/0.765/0.77, CNN 0.58/0.535/0.565, CNN-LSTM 0.055/0.075/0.165, LLM-guided 0.245/0.41/0.13.
Convergence (first episode where the 100-episode moving success rate reaches 90% of its final plateau): baseline 981 ± 95, CNN 1345 ± 183. For the two LSTM agents this metric is not meaningful because they never reached a stable plateau within the budget (see `results_table.md`).

Figures: `figures/learning_curves.png`, `figures/final_metrics.png`, `figures/trajectories.png` (same scenario per row for all four agents).

## 4. Analysis
1. **The simplest agent won in this budget.** The sensor-only baseline reached 78% success with the lowest dose and latency. It receives explicit goal-relative coordinates, so the task is easy for it and it learns in about 1000 episodes.
2. **Adding the CNN slowed learning** (56% at 2000 episodes, still rising). The CNN-DRL agent has strictly more information than the baseline, so this is a sample-efficiency effect, not proof that images are useless. A pilot on one seed showed the CNN agent going from about 35% at 1000 episodes to 58% at 2000.
3. **Adding the LSTM slowed learning further** (about 10%), consistent with recurrent Q-learning being the most data-hungry variant. Its learning curve was still rising slowly at the end, so these numbers reflect an undertrained model, not a ceiling.
4. **The advisory features helped the recurrent agent: 9.8% to 26.2% success** (all three LLM-guided seeds ended at or above 13%, versus 5.5-16.5% without; seed variance is large, 11.5 points). It also reduced travel time (30.6 to 24.1 steps) and shield interventions (27.1 to 22.9), although dose and over-limit rate were not improved. With three seeds this is suggestive evidence, not a statistically established effect.
5. **Safety:** the shield is always active, so wall collisions are impossible and cells with high dose rate cannot be entered. Over-limit failures (13-25%) come from cumulative dose along longer or wandering routes, which the shield does not prevent. The "interventions" column counts how often an agent's preferred action was blocked, a rough proxy for how often its raw policy was unsafe. The weaker agents were blocked far more often.
6. **Latency:** all agents run well under 2 ms per decision on CPU.

## 5. Limitations
- Compute-limited: 2000 episodes, 3 seeds, one 10x10 environment family. The LSTM variants need substantially more training before the comparison is a fair measure of their potential; I would expect the gap to the baseline to narrow with a longer budget, but I did not test that.
- The image-fusion design (gated image branch, narrower CNN output) was chosen after short pilot runs of the CNN agent on seed 0, which scored on the same held-out evaluation scenarios used for the final table. That is a small selection effect, applied only to the CNN-family architecture.
- The LLM is a rule-based stand-in (see Section 2). The advisory text is generated from templates by the simulator, so a real LLM would face easier parsing but a different latency and failure profile.
- The shield uses calibrated adjacent-cell dose information; real hardware would need validated sensing.
- Results are simulation-only and must not be used to justify real deployment without independent validation and authorized supervision.

## 6. Files
`env.py` environment and shield, `advisory.py` advisory interpreter, `models.py` networks, `train.py` Double-DQN training and evaluation, `run_all.py` runs all 12 jobs, `analyze.py` tables and figures. Outputs: `models/` (12 trained models), `results/` (per-run JSON with learning curves), `results_table.md`, `results.csv`, `figures/`, `logs/`.
Reproduce: `pip install torch numpy matplotlib; python run_all.py; python analyze.py`.
