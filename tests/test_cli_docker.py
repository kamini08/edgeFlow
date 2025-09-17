import sys
from types import ModuleType

import pytest

import edgeflowc


def _write_cfg(tmp_path, content: str) -> str:
    p = tmp_path / "docker.ef"
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_cli_docker_import_missing(tmp_path, monkeypatch):
    """When --docker is set but docker_manager import fails, exit with code 1."""
    cfg = _write_cfg(tmp_path, 'model="m.tflite"\n')

    # Ensure docker_manager import fails
    if "docker_manager" in sys.modules:
        del sys.modules["docker_manager"]

    monkeypatch.setattr(sys, "argv", ["edgeflowc.py", cfg, "--docker"])  # type: ignore[attr-defined]
    rc = edgeflowc.main()
    assert rc == 1


def test_cli_docker_build_and_run_success(tmp_path, monkeypatch):
    """Happy-path: docker build (optional) and run succeed -> exit 0."""
    cfg = _write_cfg(tmp_path, 'model="m.tflite"\n')

    # Fake docker_manager module
    fake = ModuleType("docker_manager")

    class FakeDM:
        def build_image(self, **kwargs):
            return True

        def run_optimization_pipeline(self, **kwargs):
            return {"success": True, "output_path": "./outputs"}

    def validate_docker_setup():
        return {
            "docker_installed": True,
            "compose_installed": True,
            "docker_running": True,
        }

    fake.DockerManager = lambda: FakeDM()  # type: ignore[attr-defined]
    fake.validate_docker_setup = validate_docker_setup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "docker_manager", fake)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "edgeflowc.py",
            cfg,
            "--docker",
            "--docker-build",
            "--docker-tag",
            "edgeflow:test",
        ],
    )  # type: ignore[attr-defined]
    rc = edgeflowc.main()
    assert rc == 0


def test_cli_docker_run_failure(tmp_path, monkeypatch):
    """If docker run returns success False, exit with code 1."""
    cfg = _write_cfg(tmp_path, 'model="m.tflite"\n')

    fake = ModuleType("docker_manager")

    class FakeDM:
        def build_image(self, **kwargs):
            return True

        def run_optimization_pipeline(self, **kwargs):
            return {"success": False, "error": "boom"}

    def validate_docker_setup():
        return {
            "docker_installed": True,
            "compose_installed": True,
            "docker_running": True,
        }

    fake.DockerManager = lambda: FakeDM()  # type: ignore[attr-defined]
    fake.validate_docker_setup = validate_docker_setup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "docker_manager", fake)

    monkeypatch.setattr(sys, "argv", ["edgeflowc.py", cfg, "--docker"])  # type: ignore[attr-defined]
    rc = edgeflowc.main()
    assert rc == 1
