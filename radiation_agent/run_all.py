"""Run all 4 agents x 3 seeds (2 processes in parallel)."""
import subprocess, sys
from concurrent.futures import ThreadPoolExecutor
EP = 2000
jobs = [(a, s) for s in range(3) for a in ("llm_cnn_lstm", "cnn_lstm", "cnn", "baseline")]
def go(j):
    a, s = j
    with open(f"logs/{a}_seed{s}.txt", "w") as f:
        subprocess.run([sys.executable, "train.py", "--agent", a, "--seed", str(s),
                        "--episodes", str(EP), "--out", "results"], stdout=f, stderr=subprocess.STDOUT)
with ThreadPoolExecutor(2) as ex:
    list(ex.map(go, jobs))
print("ALL DONE")
