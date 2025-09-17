"""
EdgeFlow Docker Management Module

Provides programmatic control over Docker containers for EdgeFlow.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Configuration for Docker container."""

    image: str
    name: str
    volumes: Dict[str, Dict[str, str]]
    environment: Dict[str, str]
    ports: Optional[Dict[str, int]] = None
    command: Optional[List[str]] = None
    network: Optional[str] = None
    healthcheck: Optional[Dict[str, Any]] = None


def _which(cmd: str) -> bool:
    return subprocess.run(["which", cmd], capture_output=True).returncode == 0


def _run(
    cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None
) -> subprocess.CompletedProcess:
    logger.debug("Running command: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


class DockerManager:
    """Manages Docker operations for EdgeFlow."""

    def __init__(self) -> None:
        """Initialize Docker client via subprocess calls (SDK optional)."""
        self.use_sdk = False
        try:
            import docker  # type: ignore

            self.docker = docker.from_env()  # type: ignore[attr-defined]
            # Ping to ensure connectivity
            self.docker.ping()  # type: ignore[attr-defined]
            self.use_sdk = True
            logger.info("Docker SDK client initialized successfully")
        except Exception as e:  # noqa: BLE001
            logger.warning("Docker SDK unavailable: %s; falling back to subprocess", e)
            self.docker = None

    def build_image(
        self,
        dockerfile_path: str = "Dockerfile",
        tag: str = "edgeflow:latest",
        build_args: Optional[Dict[str, str]] = None,
        context: str = ".",
    ) -> bool:
        """Build Docker image with proper error handling."""
        build_args = build_args or {}
        if self.use_sdk:
            try:
                image, logs = self.docker.images.build(  # type: ignore[attr-defined]
                    path=context,
                    dockerfile=dockerfile_path,
                    tag=tag,
                    buildargs=build_args,
                    rm=True,
                )
                for _ in logs:
                    pass
                return True
            except Exception as e:  # noqa: BLE001
                logger.error("Docker SDK build failed: %s", e)
                return False
        # Fallback: subprocess
        cmd = [
            "docker",
            "build",
            "-t",
            tag,
            "-f",
            dockerfile_path,
            context,
        ]
        for k, v in build_args.items():
            cmd.extend(["--build-arg", f"{k}={v}"])
        res = _run(cmd)
        if res.returncode != 0:
            logger.error("Docker build failed: %s", res.stderr)
            return False
        return True

    def run_optimization_pipeline(
        self,
        config_file: str,
        model_path: str,
        device_spec_file: Optional[str] = None,
        output_dir: str = "./outputs",
        image: str = "edgeflow:latest",
    ) -> Dict[str, Any]:
        """
        Run complete EdgeFlow optimization pipeline in Docker.

        This includes:
        1. Initial compatibility check
        2. Optimization if needed
        3. Report generation
        """
        cfg_path = Path(config_file).resolve()
        outputs = Path(output_dir).resolve()
        outputs.mkdir(parents=True, exist_ok=True)
        models = Path(model_path).resolve().parent
        specs = Path(device_spec_file).resolve().parent if device_spec_file else None

        command: List[str] = ["python", "edgeflowc.py", f"/app/configs/{cfg_path.name}"]
        if device_spec_file:
            command += [
                "--device-spec-file",
                f"/app/device_specs/{Path(device_spec_file).name}",
            ]

        volumes = {
            str(cfg_path.parent): {"bind": "/app/configs", "mode": "ro"},
            str(models): {"bind": "/app/models", "mode": "rw"},
            str(outputs): {"bind": "/app/outputs", "mode": "rw"},
        }
        if specs:
            volumes[str(specs)] = {"bind": "/app/device_specs", "mode": "ro"}

        if self.use_sdk:
            try:
                cont = self.docker.containers.run(  # type: ignore[attr-defined]
                    image,
                    command,
                    volumes=volumes,
                    remove=True,
                    detach=True,
                    environment={"PYTHONUNBUFFERED": "1"},
                )
                logs = cont.logs(stream=True)
                for _ in logs:
                    pass
                code = cont.wait()["StatusCode"]
                return {
                    "success": code == 0,
                    "output_path": str(outputs),
                    "exit_code": code,
                }
            except Exception as e:  # noqa: BLE001
                return {"success": False, "error": str(e)}

        # Subprocess fallback
        vol_flags: List[str] = []
        for host, spec in volumes.items():
            vol_flags += ["-v", f"{host}:{spec['bind']}"]
        res = _run(["docker", "run", "--rm", *vol_flags, image, *command])
        return {
            "success": res.returncode == 0,
            "output_path": str(outputs),
            "exit_code": res.returncode,
            "error": res.stderr if res.returncode != 0 else "",
        }

    def start_services(self) -> bool:
        """Start all EdgeFlow services using docker-compose."""
        res = _run(["docker-compose", "up", "-d"])
        if res.returncode != 0:
            logger.error("docker-compose up failed: %s", res.stderr)
            return False
        return True

    def stop_services(self) -> bool:
        """Stop all EdgeFlow services."""
        res = _run(["docker-compose", "down"])
        if res.returncode != 0:
            logger.error("docker-compose down failed: %s", res.stderr)
            return False
        return True

    def get_service_status(self) -> Dict[str, str]:
        """Get status of all EdgeFlow services."""
        res = _run(["docker-compose", "ps", "--services", "--filter", "status=running"])
        running = set(res.stdout.split()) if res.returncode == 0 else set()
        # Basic inference based on compose services
        services = ["edgeflow-compiler", "edgeflow-api", "edgeflow-frontend"]
        return {svc: ("running" if svc in running else "stopped") for svc in services}

    def cleanup(self, remove_images: bool = False) -> None:
        """Clean up containers and optionally images."""
        _run(["docker-compose", "down", "-v"])
        if remove_images:
            _run(["docker", "image", "prune", "-f"])


def validate_docker_setup() -> Dict[str, bool]:
    """
    Validate Docker setup and dependencies.

    Returns dict with status of each component.
    """
    status = {
        "docker_installed": _which("docker"),
        "compose_installed": _which("docker-compose") or _which("docker"),
        "docker_running": False,
    }
    if status["docker_installed"]:
        out = _run(["docker", "info"])  # lightweight daemon ping
        status["docker_running"] = out.returncode == 0
    return status
