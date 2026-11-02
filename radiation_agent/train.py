"""Train + evaluate one agent/seed with Double DQN.  Usage:
    python train.py --agent llm_cnn_lstm --seed 0 --episodes 1500
"""
import argparse, json, os, time, copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import RadiationZoneEnv, N, VEC_DIM, VEC_BASE, N_ACTIONS
from advisory import AdvisoryInterpreter, paint
from models import QNet

AGENTS = {
    "baseline":     dict(kind="baseline", K=1, llm=False),
    "cnn":          dict(kind="cnn",      K=1, llm=False),
    "cnn_lstm":     dict(kind="cnn_lstm", K=4, llm=False),
    "llm_cnn_lstm": dict(kind="cnn_lstm", K=4, llm=True),
}
GAMMA, LR, BATCH = 0.98, 7e-4, 64
UPDATE_EVERY, TARGET_SYNC, WARMUP, BUF_EPISODES = 2, 300, 600, 500
TRAIN_SEED0, EVAL_SEED0, CKPT_SEED0 = 1_000_000, 500_000, 400_000
N_EVAL, N_CKPT = 200, 20


class FrameBuilder:
    """Turns env observation into (img, vec, mask); adds advisory features for the LLM agent."""
    def __init__(self, llm):
        self.llm, self.interp = llm, AdvisoryInterpreter("rule")

    def build(self, env):
        img, vec, mask = env.obs
        v = np.zeros(VEC_DIM, np.float32)
        v[:VEC_BASE] = vec
        if self.llm:
            f = self.interp.parse(env.advisory)
            v[VEC_BASE:] = f
            img = img.copy()
            img[3] = paint(f)
        return img, v, mask


def window(imgs, vecs, t, K):
    lo = t - K + 1
    im, ve = imgs[max(lo, 0):t + 1], vecs[max(lo, 0):t + 1]
    if lo < 0:
        im = np.concatenate([np.zeros((-lo,) + im.shape[1:], im.dtype), im])
        ve = np.concatenate([np.zeros((-lo,) + ve.shape[1:], ve.dtype), ve])
    return im, ve


def masked_argmax(q, mask):
    q = q.copy()
    q[~mask] = -1e9
    return int(np.argmax(q))


def run_episode(env, net, fb, K, seed, eps, rng, measure=False):
    env.reset(seed)
    imgs, vecs, masks, acts, rews = [], [], [], [], []
    interventions, lat = 0, []
    info = {"outcome": "running", "dose": 0.0}
    while True:
        t0 = time.perf_counter()
        img, vec, mask = fb.build(env)
        imgs.append(img); vecs.append(vec); masks.append(mask)
        t = len(imgs) - 1
        if rng.random() < eps:
            a = int(rng.choice(np.flatnonzero(mask)))
        else:
            wi, wv = window(np.asarray(imgs), np.asarray(vecs), t, K)
            with torch.no_grad():
                q = net(torch.from_numpy(wi)[None], torch.from_numpy(wv)[None])[0].numpy()
            a = masked_argmax(q, mask)
            if measure:
                interventions += int(np.argmax(q) != a)
        if measure:
            lat.append((time.perf_counter() - t0) * 1000)
        r, done, info = env.step(a)
        acts.append(a); rews.append(r)
        if done:
            break
    img, vec, mask = fb.build(env)
    imgs.append(img); vecs.append(vec); masks.append(mask)
    ep = dict(imgs=np.asarray(imgs, np.float32), vecs=np.asarray(vecs, np.float32),
              masks=np.asarray(masks), acts=np.asarray(acts), rews=np.asarray(rews, np.float32))
    stats = dict(reward=float(sum(rews)), outcome=info["outcome"], exposure=float(env.exposure),
                 steps=len(acts), interventions=interventions, lat=lat)
    return ep, stats


def evaluate(env, net, fb, K, seed0, n, rng, measure=False):
    net.eval()
    S = [run_episode(env, net, fb, K, seed0 + i, 0.0, rng, measure)[1] for i in range(n)]
    net.train()
    succ = [s["outcome"] == "success" for s in S]
    out = dict(
        reward=float(np.mean([s["reward"] for s in S])),
        success=float(np.mean(succ)),
        overdose=float(np.mean([s["outcome"] == "overdose" for s in S])),
        timeout=float(np.mean([s["outcome"] == "timeout" for s in S])),
        exposure=float(np.mean([s["exposure"] for s in S])),
        travel_steps=float(np.mean([s["steps"] for s, ok in zip(S, succ) if ok])) if any(succ) else float("nan"),
        interventions=float(np.mean([s["interventions"] for s in S])),
    )
    if measure:
        allat = np.concatenate([s["lat"] for s in S])
        out["latency_ms"] = float(np.mean(allat))
        out["latency_p95_ms"] = float(np.percentile(allat, 95))
    return out


def sample_batch(buf, K, rng):
    wi, wv, ni, nv, a, r, d, nm = [], [], [], [], [], [], [], []
    for _ in range(BATCH):
        ep = buf[int(rng.integers(len(buf)))]
        T = len(ep["acts"])
        t = int(rng.integers(T))
        i0, v0 = window(ep["imgs"], ep["vecs"], t, K)
        i1, v1 = window(ep["imgs"], ep["vecs"], t + 1, K)
        wi.append(i0); wv.append(v0); ni.append(i1); nv.append(v1)
        a.append(ep["acts"][t]); r.append(ep["rews"][t]); d.append(float(t == T - 1)); nm.append(ep["masks"][t + 1])
    f = lambda x: torch.from_numpy(np.asarray(x, np.float32))
    return f(wi), f(wv), f(ni), f(nv), torch.from_numpy(np.asarray(a)).long(), f(r), f(d), torch.from_numpy(np.asarray(nm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=list(AGENTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = AGENTS[args.agent]
    K = cfg["K"]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    env, fb = RadiationZoneEnv(), FrameBuilder(cfg["llm"])
    net = QNet(cfg["kind"]); tgt = copy.deepcopy(net)
    img_scale = float(os.environ.get("IMG_LR_SCALE", "1.0"))
    img_p = [p for n, p in net.named_parameters() if n.startswith("enc.cnn") or n.startswith("enc.gate") or n.startswith("enc.ln")]
    other_p = [p for n, p in net.named_parameters() if not (n.startswith("enc.cnn") or n.startswith("enc.gate") or n.startswith("enc.ln"))]
    opt = torch.optim.Adam([dict(params=other_p, lr=LR), dict(params=img_p, lr=LR * img_scale)])
    buf = deque(maxlen=BUF_EPISODES)
    n_params = sum(p.numel() for p in net.parameters())

    log = dict(agent=args.agent, seed=args.seed, episodes=args.episodes, params=n_params,
               train_reward=[], train_success=[], train_exposure=[], ckpt=[])
    steps = updates = 0
    t_start = time.time()
    for e in range(args.episodes):
        eps = max(0.05, 1.0 - 0.95 * e / (0.6 * args.episodes))
        ep, st = run_episode(env, net, fb, K, TRAIN_SEED0 + args.seed * 10_000 + e, eps, rng)
        buf.append(ep)
        steps += st["steps"]
        log["train_reward"].append(st["reward"])
        log["train_success"].append(int(st["outcome"] == "success"))
        log["train_exposure"].append(st["exposure"])
        n_up = st["steps"] // UPDATE_EVERY if steps >= WARMUP else 0
        for _ in range(n_up):
            wi, wv, ni, nv, a, r, d, nm = sample_batch(buf, K, rng)
            with torch.no_grad():
                qn = net(ni, nv); qn[~nm] = -1e9
                a2 = qn.argmax(1, keepdim=True)
                y = r + GAMMA * (1 - d) * tgt(ni, nv).gather(1, a2).squeeze(1)
            q = net(wi, wv).gather(1, a[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(q, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            updates += 1
            if updates % TARGET_SYNC == 0:
                tgt.load_state_dict(net.state_dict())
        if (e + 1) % 100 == 0:
            ck = evaluate(env, net, fb, K, CKPT_SEED0, N_CKPT, rng)
            log["ckpt"].append(dict(episode=e + 1, success=ck["success"], reward=ck["reward"]))
            print(f"[{args.agent} s{args.seed}] ep {e+1} eps={eps:.2f} "
                  f"train_succ={np.mean(log['train_success'][-100:]):.2f} "
                  f"eval_succ={ck['success']:.2f} t={time.time()-t_start:.0f}s", flush=True)
    log["train_seconds"] = time.time() - t_start
    log["final_eval"] = evaluate(env, net, fb, K, EVAL_SEED0, N_EVAL, rng, measure=True)
    torch.save(net.state_dict(), f"models/{args.agent}_seed{args.seed}.pt")
    json.dump(log, open(f"{args.out}/{args.agent}_seed{args.seed}.json", "w"))
    print("FINAL", args.agent, args.seed, json.dumps(log["final_eval"]), flush=True)


if __name__ == "__main__":
    main()
