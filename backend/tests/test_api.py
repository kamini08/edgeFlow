from __future__ import annotations

import base64
from typing import Any, Dict  # noqa: F401 (placeholders kept for future tests)

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_version():
    r = client.get("/api/version")
    assert r.status_code == 200
    assert "version" in r.json()


def test_help():
    r = client.get("/api/help")
    assert r.status_code == 200
    data = r.json()
    assert any("/api/compile" in x for x in data["commands"])


def test_compile_success():
    cfg = 'model_path="m.tflite"\nquantize=int8\n'
    r = client.post("/api/compile", json={"config_file": cfg, "filename": "c.ef"})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["parsed_config"]["model_path"] == "m.tflite"


def test_compile_invalid_ext():
    r = client.post("/api/compile", json={"config_file": "x=1", "filename": "a.txt"})
    assert r.status_code == 400


def test_compile_verbose_logs():
    cfg = 'model_path="m.tflite"\n'
    r = client.post(
        "/api/compile/verbose", json={"config_file": cfg, "filename": "c.ef"}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert isinstance(data.get("logs"), list)


def test_optimize_and_benchmark():
    raw = b"dummy model"
    b64 = base64.b64encode(raw).decode()
    r = client.post(
        "/api/optimize",
        json={
            "model_file": b64,
            "config": {"quantize": "int8", "target_device": "raspberry_pi"},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    opt_b64 = data["optimized_model"]

    r2 = client.post(
        "/api/benchmark",
        json={
            "original_model": b64,
            "optimized_model": opt_b64,
        },
    )
    assert r2.status_code == 200
    bench = r2.json()
    assert "original_stats" in bench and "optimized_stats" in bench


def test_rate_limit():
    # Basic check: a few calls should not trigger limiter.
    for _ in range(5):
        r = client.get("/api/health")
        assert r.status_code == 200
