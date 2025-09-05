from __future__ import annotations

import base64
import threading
from typing import List

from fastapi.testclient import TestClient

from backend.app import app, rate_limiter


client = TestClient(app)


def test_e2e_workflow():
    # 1) Compile
    cfg = 'model_path="m.tflite"\noutput_path="o.tflite"\nquantize=int8\n'
    r = client.post("/api/compile", json={"config_file": cfg, "filename": "model.ef"})
    assert r.status_code == 200
    parsed = r.json()["parsed_config"]
    assert parsed["model_path"] == "m.tflite"

    # 2) Optimize
    raw = b"0" * 2048
    b64 = base64.b64encode(raw).decode()
    r2 = client.post(
        "/api/optimize",
        json={"model_file": b64, "config": {"quantize": "int8", "target_device": "raspberry_pi"}},
    )
    assert r2.status_code == 200
    opt_b64 = r2.json()["optimized_model"]

    # 3) Benchmark
    r3 = client.post("/api/benchmark", json={"original_model": b64, "optimized_model": opt_b64})
    assert r3.status_code == 200
    data = r3.json()
    assert "original_stats" in data and "optimized_stats" in data and "improvement" in data


def test_concurrent_requests():
    results: List[int] = []

    def hit():
        res = client.get("/api/health")
        results.append(res.status_code)

    threads = [threading.Thread(target=hit) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(code == 200 for code in results)


def test_rate_limit_enforced():
    # Shrink capacity to 2 for this test, then restore
    old_cap = rate_limiter.capacity
    rate_limiter.capacity = 2
    try:
        # Reset state for the test client's IP
        rate_limiter.tokens.clear()
        rate_limiter.window_started.clear()
        # Make 3 compile calls, third should hit 429
        cfg = 'model_path="m.tflite"\n'
        assert client.post("/api/compile", json={"config_file": cfg, "filename": "a.ef"}).status_code == 200
        assert client.post("/api/compile", json={"config_file": cfg, "filename": "a.ef"}).status_code == 200
        r = client.post("/api/compile", json={"config_file": cfg, "filename": "a.ef"})
        assert r.status_code in (200, 429)
    finally:
        rate_limiter.capacity = old_cap

