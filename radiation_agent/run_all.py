"""Run all 4 agents x 3 seeds (2 processes in parallel by default).

    python run_all.py                -> results/, models/, logs/          (v1 snapshot; overwrites it)
    python run_all.py --tag v2       -> results_v2/, models_v2/, logs_v2/ (new run; v1 untouched)

Then view the new run in the dashboard:
    RAD_RESULTS_DIR=results_v2 RAD_MODELS_DIR=models_v2 python -m uvicorn server:app --port 8000
"""
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

EP = 2000
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="", help="suffix for results/models/logs directories")
ap.add_argument("--workers", type=int, default=2)
args = ap.parse_args()

sfx = f"_{args.tag}" if args.tag else ""
RES, MOD, LOG = f"results{sfx}", f"models{sfx}", f"logs{sfx}"
for d in (RES, MOD, LOG):
    os.makedirs(d, exist_ok=True)

jobs = [(a, s) for s in range(3) for a in ("llm_cnn_lstm", "cnn_lstm", "cnn", "baseline")]


def go(j):
    a, s = j
    with open(f"{LOG}/{a}_seed{s}.txt", "w") as f:
        subprocess.run([sys.executable, "train.py", "--agent", a, "--seed", str(s), "--episodes", str(EP),
                        "--out", RES, "--models", MOD], stdout=f, stderr=subprocess.STDOUT)


with ThreadPoolExecutor(args.workers) as ex:
    list(ex.map(go, jobs))
print("ALL DONE ->", RES, MOD)
