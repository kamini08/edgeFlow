from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_compile_dry_run_success():
    cfg = 'model_path="m.tflite"\nquantize=int8\n'
    r = client.post(
        "/api/compile/dry-run", json={"config_file": cfg, "filename": "c.ef"}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["parsed_config"]["model_path"] == "m.tflite"


def test_compile_dry_run_invalid_ext():
    r = client.post(
        "/api/compile/dry-run", json={"config_file": "x=1", "filename": "a.txt"}
    )
    assert r.status_code == 400


def test_compile_verbose_invalid_ext():
    r = client.post(
        "/api/compile/verbose", json={"config_file": "x=1", "filename": "a.txt"}
    )
    assert r.status_code == 400


def test_compile_invalid_syntax_returns_400():
    bad = "invalid syntax ==="
    r = client.post("/api/compile", json={"config_file": bad, "filename": "c.ef"})
    assert r.status_code == 400
    assert "syntax" in r.json()["detail"].lower()


def test_compile_verbose_invalid_syntax_returns_400():
    bad = "invalid syntax ==="
    r = client.post(
        "/api/compile/verbose", json={"config_file": bad, "filename": "c.ef"}
    )
    assert r.status_code == 400
    assert "syntax" in r.json()["detail"].lower()


def test_payload_too_large_middleware():
    # Simulate a very large content-length header; middleware should return 413
    cfg = 'model_path="m.tflite"\n'
    r = client.post(
        "/api/compile",
        headers={"content-length": str(200 * 1024 * 1024)},  # 200MB
        json={"config_file": cfg, "filename": "c.ef"},
    )
    assert r.status_code == 413


def test_root_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("docs") == "/docs"


def test_help_lists_dry_run():
    r = client.get("/api/help")
    assert r.status_code == 200
    cmds = r.json()["commands"]
    assert any("/api/compile/dry-run" in c for c in cmds)
