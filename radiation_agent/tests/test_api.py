"""API integration tests. Skipped automatically when torch / fastapi are not installed."""
import pytest

try:
    import torch  # noqa: F401
except Exception as exc:  # torch can fail with non-ImportError when CUDA libraries are missing
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

client = TestClient(server.app)


def test_health_reports_run_and_models():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["simulation_only"] is True
    assert body["agents"] == ["baseline", "cnn", "cnn_lstm", "llm_cnn_lstm"]


def test_compare_rejects_out_of_range_seed():
    assert client.get("/api/compare", params={"seed": -1}).status_code == 422


def test_compare_rejects_unknown_model_seed():
    assert client.get("/api/compare", params={"seed": 500000, "model_seed": 99}).status_code in (404, 422)


def test_episode_rejects_unknown_agent():
    assert client.get("/api/episode", params={"agent": "nope"}).status_code == 400
