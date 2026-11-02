"""Simulator invariants. These must hold for every seed, otherwise reported results are not comparable."""
import hashlib
import json

import numpy as np
import pytest

from env import ACTIONS, LIMIT, MAX_STEPS, N, RadiationZoneEnv

# Fingerprints of the layouts used by the reported experiments (evaluation seeds 500000, 500003, 500199).
# If this test fails, the environment changed and the historical results in results/ no longer match it.
GOLDEN = {500000: "b9f82006da945c06", 500003: "73839bdc8cb0b281", 500199: "90819d0a44beed6d"}


def fingerprint(env):
    blob = json.dumps({"walls": env.walls.astype(int).tolist(), "start": list(env.start), "goal": list(env.goal),
                       "sources": [[int(x[0]), int(x[1]), round(x[2], 6)] for x in env.sources]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@pytest.mark.parametrize("seed", sorted(GOLDEN))
def test_evaluation_layouts_match_reported_experiments(seed):
    env = RadiationZoneEnv()
    env.reset(seed)
    assert fingerprint(env) == GOLDEN[seed]


def test_reset_is_deterministic_and_seed_sensitive():
    a, b, c = RadiationZoneEnv(), RadiationZoneEnv(), RadiationZoneEnv()
    a.reset(42); b.reset(42); c.reset(43)
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c)


@pytest.mark.parametrize("seed", range(500000, 500020))
def test_start_and_goal_are_reachable_and_free(seed):
    env = RadiationZoneEnv()
    env.reset(seed)
    assert not env.walls[env.start] and not env.walls[env.goal]
    assert env._reachable(env.walls, env.start, env.goal)


@pytest.mark.parametrize("seed", range(500000, 500010))
def test_shield_mask_always_allows_wait_and_never_allows_walls(seed):
    env = RadiationZoneEnv()
    env.reset(seed)
    rng = np.random.default_rng(seed)
    for _ in range(MAX_STEPS):
        mask = env.shield_mask()
        assert mask[4]  # waiting is always permitted
        for i, (dr, dc) in enumerate(ACTIONS[:4]):
            rr, cc = env.pos[0] + dr, env.pos[1] + dc
            if mask[i]:
                assert 0 <= rr < N and 0 <= cc < N and not env.walls[rr, cc]
        _, done, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
        if done:
            break


def test_disallowed_action_is_replaced_by_wait():
    """The shield lives in the environment too: a forbidden move must not change the position."""
    found = None
    for seed in range(500000, 500200):
        env = RadiationZoneEnv()
        env.reset(seed)
        for _ in range(MAX_STEPS):
            mask = env.shield_mask()
            bad = [i for i in range(4) if not mask[i] and 0 <= env.pos[0] + ACTIONS[i][0] < N
                   and 0 <= env.pos[1] + ACTIONS[i][1] < N and not env.walls[env.pos[0] + ACTIONS[i][0], env.pos[1] + ACTIONS[i][1]]]
            if bad:
                found = (env, bad[0])
                break
            ok = np.flatnonzero(mask)
            _, done, _ = env.step(int(ok[0]))
            if done:
                break
        if found:
            break
    assert found is not None, "expected at least one hazard-blocked move in the evaluation seeds"
    env, action = found
    before = env.pos
    _, _, info = env.step(action)
    assert env.pos == before
    assert info["shield_override"] is True


@pytest.mark.parametrize("seed", range(500000, 500030))
def test_masked_actions_are_never_overridden(seed):
    env = RadiationZoneEnv()
    env.reset(seed)
    rng = np.random.default_rng(seed)
    for _ in range(MAX_STEPS):
        _, done, info = env.step(int(rng.choice(np.flatnonzero(env.shield_mask()))))
        assert info["shield_override"] is False
        if done:
            break


@pytest.mark.parametrize("seed", range(500000, 500010))
def test_episode_ends_with_a_valid_outcome_within_step_limit(seed):
    env = RadiationZoneEnv()
    env.reset(seed)
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        _, done, info = env.step(4)  # always wait: must end by timeout or dose overrun
        if done:
            break
    assert done
    assert info["outcome"] in {"success", "overdose", "timeout"}
    assert steps <= MAX_STEPS
    if info["outcome"] == "overdose":
        assert env.exposure >= LIMIT


@pytest.mark.parametrize("seed", range(500000, 500010))
def test_sources_stay_on_free_cells_while_drifting(seed):
    env = RadiationZoneEnv()
    env.reset(seed)
    for _ in range(MAX_STEPS):
        for r, c, strength, _ in env.sources:
            assert 0 <= r < N and 0 <= c < N and not env.walls[r, c]
            assert 4.0 <= strength <= 7.0
        env._drift()


def test_observation_shapes_and_ranges():
    env = RadiationZoneEnv()
    img, vec, mask = env.reset(500000)
    assert img.shape == (4, N, N) and vec.shape == (15,) and mask.shape == (5,)
    assert np.isfinite(img).all() and np.isfinite(vec).all()
    assert mask.dtype == bool
