"""The advisory parser is a deterministic rule-based stand-in for an LLM. Its outputs must stay bounded."""
import numpy as np

from advisory import AdvisoryInterpreter, paint
from env import N


def test_parses_levels_into_quadrant_features():
    text = ("ADVISORY T+05: strong radiation reported in the north-west sector, source drifting north; "
            "weak radiation reported in the south-east sector, source stationary.")
    f = AdvisoryInterpreter("rule").parse(text)
    assert f.shape == (5,)
    assert f[0] == 1.0 and f[3] == 0.3          # north-west strong, south-east weak
    assert f[1] == 0.0 and f[2] == 0.0          # unmentioned sectors stay zero
    assert f[4] == 0.0                          # no caution flag


def test_takes_the_maximum_when_a_sector_is_reported_twice():
    text = ("moderate radiation reported in the north-east sector; strong radiation reported in the north-east sector")
    assert AdvisoryInterpreter("rule").parse(text)[1] == 1.0


def test_warning_sets_caution_flag_only():
    f = AdvisoryInterpreter("rule").parse("ADVISORY T+10: WARNING: dose budget 60% used")
    assert f[4] == 1.0 and f[:4].sum() == 0.0


def test_ambiguous_or_garbage_text_yields_zero_features():
    interp = AdvisoryInterpreter("rule")
    for text in ["", "radiation levels unknown", "strong radiation reported in the centre sector", "??? 42"]:
        assert np.allclose(interp.parse(text), 0.0)


def test_features_are_always_bounded():
    interp = AdvisoryInterpreter("rule")
    f = interp.parse("strong radiation reported in the north-west sector " * 20 + " WARNING")
    assert ((f >= 0.0) & (f <= 1.0)).all()


def test_llm_backend_output_is_clipped_to_the_safe_range():
    fake = lambda text: {"hazard": [5.0, -3.0, 0.5, 2.0], "caution": 9.0}
    f = AdvisoryInterpreter("llm", llm_fn=fake).parse("anything")
    assert f.tolist() == [1.0, 0.0, 0.5, 1.0, 1.0]


def test_paint_builds_a_quadrant_map():
    m = paint(np.array([1.0, 0.0, 0.0, 0.5, 0.0], np.float32))
    assert m.shape == (N, N)
    assert m[0, 0] == 1.0 and m[N - 1, N - 1] == 0.5 and m[0, N - 1] == 0.0
