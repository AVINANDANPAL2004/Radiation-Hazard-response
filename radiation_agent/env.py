"""Simulated radiation-hazard facility (SIMULATION ONLY).

A robot must travel from the facility entrance (bottom-left) to an inspection
point (top-right) through a partially known maze of walls, while two drifting
radiation sources make some corridors dangerous.

Observations (all noisy / partial):
  * image  (4, N, N): ch0 local radiation readings (only within SENSE_R cells),
                      ch1 wall blueprint, ch2 agent(1.0)/goal(0.5),
                      ch3 advisory hazard map (filled only for the LLM-guided agent)
  * vector (15,)    : position, goal, dosimeter, 4 neighbour readings,
                      4 neighbour wall flags, exposure fraction, time fraction
  * text advisory   : natural-language radiation report, refreshed every 5 steps
                      (15 % of reports name a wrong sector -> noisy information)

Safety layer ("shield"): a rule-based mask that forbids moves into walls and
into cells whose per-step dose would exceed R_HARD. It is applied identically to
every agent, and nothing (including the LLM features) can override it.
"""
import numpy as np
from collections import deque

N = 10
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up, down, left, right, wait
ACTION_NAMES = ["up", "down", "left", "right", "wait"]
LIMIT = 1.0          # lifetime dose budget (normalised units); exceeding = failure
K_DOSE = 0.04        # dose per step = K_DOSE * dose-rate of the occupied cell
R_HARD = 0.20        # shield blocks entering cells with per-step dose above this
SENSE_R = 2          # radiation sensor radius (cells)
MAX_STEPS = 50
RATE_NORM = 7.0
QUADS = ["north-west", "north-east", "south-west", "south-east"]
HEAD_TXT = {(-1, 0): "north", (1, 0): "south", (0, -1): "west", (0, 1): "east"}
VEC_BASE = 15
ADV_DIM = 5
VEC_DIM = VEC_BASE + ADV_DIM
N_ACTIONS = 5


def quadrant(r, c):
    return (0 if r < N // 2 else 2) + (0 if c < N // 2 else 1)


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class RadiationZoneEnv:
    def __init__(self):
        self.obs = None

    # ------------------------------------------------------------------ reset
    def reset(self, seed):
        ss = np.random.SeedSequence(int(seed))
        r_layout, r_drift, r_obs, r_adv = [np.random.default_rng(s) for s in ss.spawn(4)]
        self.r_drift, self.r_obs, self.r_adv = r_drift, r_obs, r_adv
        rng = r_layout
        while True:
            walls = rng.random((N, N)) < 0.12
            start = (N - 1 - int(rng.integers(0, 2)), int(rng.integers(0, 2)))
            goal = (int(rng.integers(0, 2)), N - 1 - int(rng.integers(0, 2)))
            walls[start] = False
            walls[goal] = False
            if self._reachable(walls, start, goal):
                break
        free = [(r, c) for r in range(N) for c in range(N)
                if not walls[r, c] and _cheb((r, c), start) >= 3 and _cheb((r, c), goal) >= 3]
        while True:
            i, j = rng.choice(len(free), 2, replace=False)
            if _cheb(free[i], free[j]) >= 3:
                break
        self.sources = [[free[k][0], free[k][1], float(rng.uniform(4.0, 7.0)), (0, 0)] for k in (i, j)]
        self.walls, self.start, self.goal = walls, start, goal
        self.pos = start
        self.exposure = 0.0
        self.t = 0
        self._compute_rate()
        self.advisory = self._make_advisory()
        self.obs = self._observe()
        return self.obs

    # ------------------------------------------------------------------- step
    def step(self, a):
        dr, dc = ACTIONS[a]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if not (0 <= nr < N and 0 <= nc < N) or self.walls[nr, nc]:
            nr, nc = self.pos  # cannot happen when the shield is respected
        prev_d = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
        self.pos = (nr, nc)
        dose = K_DOSE * float(self.rate[nr, nc])
        self.exposure += dose
        self.t += 1
        new_d = abs(nr - self.goal[0]) + abs(nc - self.goal[1])
        reward = -0.02 + 0.1 * (prev_d - new_d) - 6.0 * dose
        done, outcome = False, "running"
        if self.pos == self.goal:
            reward += 10.0
            done, outcome = True, "success"
        elif self.exposure >= LIMIT:
            reward -= 5.0
            done, outcome = True, "overdose"
        elif self.t >= MAX_STEPS:
            done, outcome = True, "timeout"
        self._drift()
        if self.t % 5 == 0:
            self.advisory = self._make_advisory()
        self.obs = self._observe()
        return reward, done, {"outcome": outcome, "dose": dose}

    # ---------------------------------------------------------------- shield
    def shield_mask(self):
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[4] = True  # waiting is always permitted
        for i, (dr, dc) in enumerate(ACTIONS[:4]):
            rr, cc = self.pos[0] + dr, self.pos[1] + dc
            if 0 <= rr < N and 0 <= cc < N and not self.walls[rr, cc] \
                    and K_DOSE * self.rate[rr, cc] <= R_HARD:
                m[i] = True
        return m

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _reachable(walls, start, goal):
        seen, dq = {start}, deque([start])
        while dq:
            r, c = dq.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in ACTIONS[:4]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N and not walls[rr, cc] and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    dq.append((rr, cc))
        return False

    def _compute_rate(self):
        rr, cc = np.mgrid[0:N, 0:N]
        rate = np.zeros((N, N))
        for r, c, inten, _ in self.sources:
            rate += inten * np.exp(-np.sqrt((rr - r) ** 2 + (cc - c) ** 2) / 1.5)
        self.rate = rate

    def _drift(self):
        # fixed number of random draws per step -> drift is identical for every agent
        for s in self.sources:
            u = self.r_drift.random()
            d = int(self.r_drift.integers(0, 4))
            if u < 0.2:
                dr, dc = ACTIONS[d]
                rr, cc = s[0] + dr, s[1] + dc
                if 0 <= rr < N and 0 <= cc < N and not self.walls[rr, cc]:
                    s[0], s[1], s[3] = rr, cc, (dr, dc)
        self._compute_rate()

    def _make_advisory(self):
        parts = []
        for r, c, inten, head in self.sources:
            q = quadrant(r, c) if self.r_adv.random() < 0.85 else int(self.r_adv.integers(0, 4))
            lvl = "strong" if inten >= 6 else "moderate" if inten >= 5 else "weak"
            drift = f"source drifting {HEAD_TXT[head]}" if head in HEAD_TXT else "source stationary"
            parts.append(f"{lvl} radiation reported in the {QUADS[q]} sector, {drift}")
        txt = f"ADVISORY T+{self.t:02d}: " + "; ".join(parts) + "."
        if self.exposure > 0.5:
            txt += f" WARNING: dose budget {int(100 * self.exposure / LIMIT)}% used, take the lowest-dose route."
        return txt

    def _observe(self):
        noise = self.r_obs.normal(0, 0.03, (N, N)).astype(np.float32)
        read = np.clip(self.rate / RATE_NORM + noise, 0, None).astype(np.float32)
        r, c = self.pos
        img = np.zeros((4, N, N), np.float32)
        r0, r1 = max(0, r - SENSE_R), min(N, r + SENSE_R + 1)
        c0, c1 = max(0, c - SENSE_R), min(N, c + SENSE_R + 1)
        img[0, r0:r1, c0:c1] = read[r0:r1, c0:c1]
        img[1] = self.walls
        img[2, self.goal[0], self.goal[1]] = 0.5
        img[2, r, c] = 1.0
        vec = np.zeros(VEC_BASE, np.float32)
        vec[0:2] = (r / N, c / N)
        vec[2:4] = (self.goal[0] / N, self.goal[1] / N)
        vec[4] = read[r, c]
        for i, (dr, dc) in enumerate(ACTIONS[:4]):
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N and not self.walls[rr, cc]:
                vec[5 + i] = read[rr, cc]
            else:
                vec[9 + i] = 1.0
        vec[13] = self.exposure / LIMIT
        vec[14] = self.t / MAX_STEPS
        return img, vec, self.shield_mask()
