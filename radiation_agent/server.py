"""FastAPI backend: live inference with the trained agents + results API + static frontend.

Run:  uvicorn server:app --port 8000        (serves the built frontend at / if frontend/dist exists)

Endpoints
  GET /api/health                      server status, available models
  GET /api/compare?seed=&model_seed=   one scenario, all four agents, LIVE inference
  GET /api/episode?agent=&seed=&model_seed=   one agent on one scenario
  GET /api/summary                     final-evaluation metrics (mean/std/per-seed)
  GET /api/curves                      learning curves (mean/std over seeds)

SIMULATION ONLY. The language component is a rule-based advisory parser, not a real LLM.
The safety shield is enforced in RadiationZoneEnv.step(): a disallowed action is replaced by wait.
"""
import glob, json, os, time
from functools import lru_cache

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from env import RadiationZoneEnv, MAX_STEPS, ACTION_NAMES
from models import QNet
import train as T

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
# choose a run with env vars, e.g. RAD_MODELS_DIR=models_v2 RAD_RESULTS_DIR=results_v2 (relative to this folder)
MODELS_DIR = os.path.join(HERE, os.environ.get("RAD_MODELS_DIR", "models"))
RESULTS_DIR = os.path.join(HERE, os.environ.get("RAD_RESULTS_DIR", "results"))
ORDER = ["baseline", "cnn", "cnn_lstm", "llm_cnn_lstm"]
MAX_SEED = 2 ** 31 - 1

app = FastAPI(title="Radiation-zone agent API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET"], allow_headers=["*"])


def model_seeds():
    return sorted({int(os.path.basename(p).split("seed")[1].split(".")[0])
                   for p in glob.glob(f"{MODELS_DIR}/*_seed*.pt")})


@lru_cache(maxsize=16)
def load_model(agent, model_seed):
    path = f"{MODELS_DIR}/{agent}_seed{model_seed}.pt"
    if not os.path.exists(path):
        raise HTTPException(404, f"No trained model for {agent} seed {model_seed}")
    cfg = T.AGENTS[agent]
    net = QNet(cfg["kind"])
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net, cfg


def _r(x):
    return np.round(np.asarray(x, float), 2).tolist()


@lru_cache(maxsize=64)
def scenario_payload(seed):
    """Layout + dose-rate maps over time. Source drift does not depend on the agent's actions,
    so one set of maps is valid for every agent."""
    env = RadiationZoneEnv()
    env.reset(seed)
    rates, sources = [], []
    for _ in range(MAX_STEPS + 1):
        rates.append(_r(env.rate.flatten()))
        sources.append([[int(s[0]), int(s[1]), round(s[2], 1), [int(s[3][0]), int(s[3][1])]] for s in env.sources])
        env._drift()
    return dict(seed=seed, walls=env.walls.astype(int).flatten().tolist(), start=list(env.start),
                goal=list(env.goal), rates=rates, sources=sources)


def rollout(agent, seed, model_seed):
    net, cfg = load_model(agent, model_seed)
    fb, K = T.FrameBuilder(cfg["llm"]), cfg["K"]
    env = RadiationZoneEnv()
    env.reset(seed)
    hi, hv, frames, lat = [], [], [], []
    while True:
        t0 = time.perf_counter()
        img, vec, mask = fb.build(env)
        hi.append(img); hv.append(vec)
        wi, wv = T.window(np.asarray(hi), np.asarray(hv), len(hi) - 1, K)
        with torch.no_grad():
            q = net(torch.from_numpy(wi)[None], torch.from_numpy(wv)[None])[0].numpy()
        act = T.masked_argmax(q, mask)          # the shield mask is applied here
        lat.append((time.perf_counter() - t0) * 1000)
        frames.append(dict(pos=list(env.pos), exposure=round(env.exposure, 3), advisory=env.advisory,
                           q=[round(float(v), 2) for v in q], mask=[bool(m) for m in mask],
                           action=int(act), blocked=bool(np.argmax(q) != act)))
        _, done, info = env.step(act)
        if done:
            break
    frames.append(dict(pos=list(env.pos), exposure=round(env.exposure, 3), advisory=env.advisory))
    return dict(outcome=info["outcome"], exposure=round(env.exposure, 3), steps=len(frames) - 1,
                latency_ms=round(float(np.mean(lat)), 3), frames=frames)


def _check(seed, model_seed):
    if not 0 <= seed <= MAX_SEED:
        raise HTTPException(400, "seed out of range")
    if model_seed not in model_seeds():
        raise HTTPException(404, f"model_seed must be one of {model_seeds()}")


@app.get("/api/health")
def health():
    return dict(status="ok", run=os.path.basename(RESULTS_DIR), agents=ORDER, model_seeds=model_seeds(), actions=ACTION_NAMES,
                eval_seeds=[T.EVAL_SEED0, T.EVAL_SEED0 + T.N_EVAL - 1], simulation_only=True)


@lru_cache(maxsize=256)
def _compare(seed, model_seed):
    # Deterministic for a given (seed, model_seed), so memoise: repeat requests skip CPU inference.
    t0 = time.perf_counter()
    scen = dict(scenario_payload(seed))
    scen["runs"] = {a: rollout(a, seed, model_seed) for a in ORDER}
    scen["model_seed"] = model_seed
    scen["compute_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return scen


@app.get("/api/compare")
def compare(seed: int = Query(T.EVAL_SEED0, ge=0, le=MAX_SEED), model_seed: int = Query(0, ge=0, le=99)):
    _check(seed, model_seed)
    return _compare(seed, model_seed)


@app.get("/api/episode")
def episode(agent: str, seed: int = Query(T.EVAL_SEED0, ge=0, le=MAX_SEED), model_seed: int = Query(0, ge=0, le=99)):
    if agent not in ORDER:
        raise HTTPException(400, f"agent must be one of {ORDER}")
    _check(seed, model_seed)
    return dict(scenario=scenario_payload(seed), agent=agent, run=rollout(agent, seed, model_seed))


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _results():
    return {a: [_load_json(f) for f in sorted(glob.glob(f"{RESULTS_DIR}/{a}_seed*.json"))] for a in ORDER}


def _ma(x, w=100):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0))
    return np.array([(c[i + 1] - c[max(0, i - w + 1)]) / (i + 1 - max(0, i - w + 1)) for i in range(len(x))])


@lru_cache(maxsize=1)
def _summary():
    R, out = _results(), {}
    for a in ORDER:
        if not R[a]:
            raise HTTPException(503, "results/ is empty; run training first")
        fe = [r["final_eval"] for r in R[a]]
        out[a] = {k: dict(mean=float(np.nanmean([f[k] for f in fe])), std=float(np.nanstd([f[k] for f in fe])),
                          per_seed=[float(f[k]) for f in fe]) for k in fe[0]}
        out[a]["params"] = R[a][0]["params"]
        out[a]["train_seconds"] = float(np.mean([r["train_seconds"] for r in R[a]]))
        out[a]["best_episode"] = dict(per_seed=[r.get("best_episode") for r in R[a]])
    return out


@lru_cache(maxsize=1)
def _curves():
    R, out = _results(), {}
    for a in ORDER:
        idx = list(range(0, len(R[a][0]["train_reward"]), 20))
        out[a] = {}
        for key in ("train_reward", "train_success"):
            M = np.array([_ma(r[key]) for r in R[a]])
            out[a][key] = dict(x=[i + 1 for i in idx], mean=_r(M.mean(0)[idx]), std=_r(M.std(0)[idx]))
        C = np.array([[c["success"] for c in r["ckpt"]] for r in R[a]])
        out[a]["eval_success"] = dict(x=[c["episode"] for c in R[a][0]["ckpt"]], mean=_r(C.mean(0)), std=_r(C.std(0)))
    return out


@app.get("/api/summary")
def summary():
    return _summary()


@app.get("/api/curves")
def curves():
    return _curves()


# ---- serve the built React app (production mode)
DIST = os.path.join(HERE, "frontend", "dist")
if os.path.isdir(DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(DIST, "index.html"))
