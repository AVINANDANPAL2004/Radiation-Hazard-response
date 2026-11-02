"""Aggregate results: table (md + csv), learning curves, bar charts, trajectories."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from env import RadiationZoneEnv, N, LIMIT
from models import QNet
import train as T

ORDER = ["baseline", "cnn", "cnn_lstm", "llm_cnn_lstm"]
LABEL = {"baseline": "Baseline DRL", "cnn": "CNN-DRL", "cnn_lstm": "CNN-LSTM-DRL",
         "llm_cnn_lstm": "LLM-guided CNN-LSTM-DRL"}
COL = {"baseline": "#7f7f7f", "cnn": "#1f77b4", "cnn_lstm": "#2ca02c", "llm_cnn_lstm": "#d62728"}
os.makedirs("figures", exist_ok=True)


def ma(x, w=100):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def convergence_episode(succ, w=100):
    """First episode where the 100-ep moving success rate reaches 90 % of its final plateau
    (mean of the last 100 episodes). NaN if the plateau is ~0."""
    m = ma(succ, w)
    final = np.mean(succ[-100:])
    if final < 0.05:
        return float("nan")
    idx = np.flatnonzero(m >= 0.9 * final)
    return float(idx[0] + 1) if len(idx) else float("nan")


R = {a: [json.load(open(f)) for f in sorted(glob.glob(f"results/{a}_seed*.json"))] for a in ORDER}
R = {a: v for a, v in R.items() if v}

metrics = [("success", "Success rate", 100, "%"), ("reward", "Mean eval reward", 1, ""),
           ("exposure", "Mean exposure (dose budget = 1.0)", 1, ""), ("overdose", "Over-limit rate", 100, "%"),
           ("travel_steps", "Travel time (steps, successes)", 1, ""),
           ("interventions", "Shield interventions / episode", 1, ""),
           ("latency_ms", "Inference latency (ms/decision)", 1, "")]
rows, csv = [], ["agent,n_seeds,metric,mean,std"]
for a in R:
    fe = [r["final_eval"] for r in R[a]]
    row = {"agent": LABEL[a], "seeds": len(fe)}
    for k, name, mult, unit in metrics:
        v = np.array([f[k] for f in fe], float) * mult
        row[k] = f"{np.nanmean(v):.2f} ± {np.nanstd(v):.2f}"
        csv.append(f"{a},{len(fe)},{k},{np.nanmean(v):.4f},{np.nanstd(v):.4f}")
    conv = np.array([convergence_episode(r["train_success"]) for r in R[a]])
    row["convergence"] = (f"{np.nanmean(conv):.0f} ± {np.nanstd(conv):.0f}" if not np.all(np.isnan(conv)) else "n/a")
    csv.append(f"{a},{len(fe)},convergence_episode,{np.nanmean(conv) if not np.all(np.isnan(conv)) else float('nan'):.1f},"
               f"{np.nanstd(conv) if not np.all(np.isnan(conv)) else float('nan'):.1f}")
    row["train_min"] = f"{np.mean([r['train_seconds'] for r in R[a]]) / 60:.1f}"
    row["params"] = R[a][0]["params"]
    rows.append(row)

hdr = ["Agent", "Seeds", "Success %", "Eval reward", "Exposure", "Over-limit %", "Travel (steps)",
       "Shield int./ep", "Latency ms", "Converge (ep)", "Train min", "Params"]
keys = ["agent", "seeds", "success", "reward", "exposure", "overdose", "travel_steps",
        "interventions", "latency_ms", "convergence", "train_min", "params"]
md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n"
for r in rows:
    md += "| " + " | ".join(str(r[k]) for k in keys) + " |\n"
open("results_table.md", "w").write(md)
open("results.csv", "w").write("\n".join(csv))
print(md)

# per-seed success (shows variance honestly)
print("per-seed final success:")
for a in R:
    print(" ", LABEL[a], [round(r["final_eval"]["success"], 3) for r in R[a]])

# --- learning curves
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for a in R:
    for j, (key, ttl) in enumerate([("train_reward", "Training reward (100-ep moving avg)"),
                                    ("train_success", "Training success rate (100-ep moving avg)")]):
        M = np.array([ma(r[key]) for r in R[a]])
        x = np.arange(1, M.shape[1] + 1)
        ax[j].plot(x, M.mean(0), color=COL[a], label=LABEL[a])
        ax[j].fill_between(x, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=COL[a], alpha=0.15)
        ax[j].set_title(ttl); ax[j].set_xlabel("Episode")
    C = np.array([[c["success"] for c in r["ckpt"]] for r in R[a]])
    xc = [c["episode"] for c in R[a][0]["ckpt"]]
    ax[2].plot(xc, C.mean(0), color=COL[a], marker="o", ms=3, label=LABEL[a])
    ax[2].fill_between(xc, C.mean(0) - C.std(0), C.mean(0) + C.std(0), color=COL[a], alpha=0.15)
ax[2].set_title("Greedy success on 20 fixed held-out scenarios"); ax[2].set_xlabel("Episode")
ax[0].legend(fontsize=8); [x.grid(alpha=0.3) for x in ax]
plt.tight_layout(); plt.savefig("figures/learning_curves.png", dpi=140); plt.close()

# --- bar charts of final metrics
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
for j, (k, name, mult, unit) in enumerate([m for m in metrics if m[0] in ("success", "exposure", "travel_steps", "latency_ms")]):
    mu = [np.nanmean([r["final_eval"][k] for r in R[a]]) * mult for a in R]
    sd = [np.nanstd([r["final_eval"][k] for r in R[a]]) * mult for a in R]
    ax[j].bar(range(len(R)), mu, yerr=sd, color=[COL[a] for a in R], capsize=3)
    ax[j].set_xticks(range(len(R))); ax[j].set_xticklabels(["Base", "CNN", "CNN-LSTM", "LLM-CL"][:len(R)], fontsize=8)
    ax[j].set_title(name, fontsize=9); ax[j].grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig("figures/final_metrics.png", dpi=140); plt.close()

# --- trajectories on identical scenarios (seed-0 models)
def load(a):
    cfg = T.AGENTS[a]
    net = QNet(cfg["kind"]); net.load_state_dict(torch.load(f"models/{a}_seed0.pt")); net.eval()
    return net, T.FrameBuilder(cfg["llm"]), cfg["K"]

def trace(a, seed):
    net, fb, K = load(a)
    env = RadiationZoneEnv(); env.reset(seed)
    hi, hv, path = [], [], [env.pos]
    while True:
        img, vec, mask = fb.build(env); hi.append(img); hv.append(vec)
        wi, wv = T.window(np.asarray(hi), np.asarray(hv), len(hi) - 1, K)
        with torch.no_grad():
            q = net(torch.from_numpy(wi)[None], torch.from_numpy(wv)[None])[0].numpy()
        _, done, info = env.step(T.masked_argmax(q, mask)); path.append(env.pos)
        if done: break
    return env, path, info["outcome"]

if all(os.path.exists(f"models/{a}_seed0.pt") for a in R):
    seeds = [T.EVAL_SEED0 + i for i in (3, 11, 25)]
    fig, ax = plt.subplots(len(seeds), len(R), figsize=(3.6 * len(R), 3.6 * len(seeds)))
    for i, s in enumerate(seeds):
        for j, a in enumerate(R):
            env, path, out = trace(a, s)
            ax[i, j].imshow(env.rate, cmap="YlOrRd", vmin=0, vmax=7)
            wr, wc = np.nonzero(env.walls); ax[i, j].scatter(wc, wr, marker="s", s=170, color="k")
            ax[i, j].plot([p[1] for p in path], [p[0] for p in path], "-o", ms=3, color="#1f4e9c", lw=1.5)
            ax[i, j].scatter(*env.goal[::-1], marker="*", s=200, color="lime", edgecolor="k", zorder=5)
            ax[i, j].scatter(*env.start[::-1], marker="D", s=60, color="cyan", edgecolor="k", zorder=5)
            ax[i, j].set_title(f"{LABEL[a].replace('LLM-guided CNN-LSTM-DRL','LLM-CNN-LSTM')}\n{out}, dose {env.exposure:.2f}", fontsize=8)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    plt.suptitle("Same scenario per row (final sources shown; blue = path, black = walls)", fontsize=9)
    plt.tight_layout(); plt.savefig("figures/trajectories.png", dpi=130); plt.close()
print("figures written")
