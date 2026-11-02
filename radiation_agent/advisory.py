"""Advisory interpreter: natural-language radiation advisory -> hazard features.

backend="rule": deterministic regex parser (offline stand-in for the LLM; this is
                what the reported experiments use).
backend="llm"  : pass `llm_fn(text) -> {"hazard": [4 floats], "caution": float}`
                to plug in a real LLM (e.g. an API call). Not used in the
                reported results.

SAFETY: the interpreter only produces bounded features in [0, 1]. They are
appended to the agent's observation. They never choose actions and never touch
the safety shield.
"""
import re
import numpy as np
from env import QUADS, N

_PAT = re.compile(r"(strong|moderate|weak) radiation reported in the "
                  r"(north-west|north-east|south-west|south-east) sector")
_LEVEL = {"strong": 1.0, "moderate": 0.6, "weak": 0.3}


class AdvisoryInterpreter:
    def __init__(self, backend="rule", llm_fn=None):
        self.backend, self.llm_fn, self._cache = backend, llm_fn, {}

    def parse(self, text):
        if text in self._cache:
            return self._cache[text]
        if self.backend == "llm":
            out = self.llm_fn(text)
            f = np.array(list(out["hazard"]) + [out["caution"]], np.float32)
        else:
            f = np.zeros(5, np.float32)
            for lvl, quad in _PAT.findall(text):
                i = QUADS.index(quad)
                f[i] = max(f[i], _LEVEL[lvl])
            f[4] = 1.0 if "WARNING" in text else 0.0
        f = np.clip(f, 0.0, 1.0)
        if len(self._cache) > 5000:
            self._cache.clear()
        self._cache[text] = f
        return f


def paint(feats):
    """Coarse (N,N) hazard map from the 4 quadrant hazard levels."""
    m = np.zeros((N, N), np.float32)
    h = N // 2
    m[:h, :h], m[:h, h:], m[h:, :h], m[h:, h:] = feats[0], feats[1], feats[2], feats[3]
    return m
