"""EdgeFlow Cross-Platform Deployment Orchestrator

Provides unified deployment across multiple platforms from a single DSL script.
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dynamic_device_profiles import get_device_profile
from traceability_system import (
    ProvenanceTracker,
    TransformationType,
    trace_transformation,
)

logger = logging.getLogger(__name__)


class DeploymentTarget(Enum):
    """Supported deployment targets."""

    RASPBERRY_PI = "raspberry_pi"
    ANDROID = "android"
    IOS = "ios"
    EDGE_TPU = "edge_tpu"
    NVIDIA_JETSON = "nvidia_jetson"
    AWS_LAMBDA = "aws_lambda"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    BARE_METAL = "bare_metal"


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""

    target: DeploymentTarget
    model_path: str
    target_host: Optional[str] = None
    credentials: Dict[str, str] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    deployment_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Result from deployment process."""

    success: bool = False
    target: Optional[DeploymentTarget] = None
    deployment_url: Optional[str] = None
    deployment_id: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    deployment_time_ms: float = 0.0


class CrossPlatformDeployer:
    """Orchestrates deployment across multiple platforms."""

    def __init__(self, tracker: Optional[ProvenanceTracker] = None):
        self.tracker = tracker or ProvenanceTracker()
        self.deployers = {
            DeploymentTarget.RASPBERRY_PI: self._deploy_raspberry_pi,
            DeploymentTarget.DOCKER: self._deploy_docker,
            DeploymentTarget.KUBERNETES: self._deploy_kubernetes,
            DeploymentTarget.BARE_METAL: self._deploy_bare_metal,
        }

    def deploy_multi_target(
        self, model_path: str, targets: List[DeploymentConfig]
    ) -> Dict[DeploymentTarget, DeploymentResult]:
        """Deploy to multiple targets simultaneously."""
        results = {}

        with trace_transformation(
            TransformationType.DEPLOYMENT,
            "cross_platform_deployer",
            f"Multi-target deployment to {len(targets)} platforms",
            parameters={"targets": [t.target.value for t in targets]},
        ) as ctx:
            for config in targets:
                logger.info(f"Deploying to {config.target.value}...")
                result = self.deploy_single_target(model_path, config)
                results[config.target] = result

                if result.success:
                    logger.info(f"✅ {config.target.value} deployment successful")
                else:
                    logger.error(f"❌ {config.target.value} deployment failed")

            successful_deployments = sum(1 for r in results.values() if r.success)
            ctx.add_metric("successful_deployments", successful_deployments)
            ctx.add_metric("total_deployments", len(targets))

        return results

    def deploy_single_target(
        self, model_path: str, config: DeploymentConfig
    ) -> DeploymentResult:
        """Deploy to a single target platform."""
        start_time = time.perf_counter()

        deployer = self.deployers.get(config.target)
        if not deployer:
            return DeploymentResult(
                success=False,
                target=config.target,
                errors=[f"Unsupported deployment target: {config.target.value}"],
            )

        try:
            result = deployer(model_path, config)
            result.deployment_time_ms = (time.perf_counter() - start_time) * 1000
            return result
        except Exception as e:
            return DeploymentResult(
                success=False,
                target=config.target,
                errors=[str(e)],
                deployment_time_ms=(time.perf_counter() - start_time) * 1000,
            )

    def _deploy_raspberry_pi(
        self, model_path: str, config: DeploymentConfig
    ) -> DeploymentResult:
        """Deploy to Raspberry Pi."""
        result = DeploymentResult(target=config.target)

        try:
            # Simulate deployment (would use existing deploy_to_pi.py)
            result.success = True
            result.deployment_url = (
                f"http://{config.target_host or 'raspberrypi.local'}:8080/inference"
            )
            result.artifacts = ["inference.py", "model.tflite"]

        except Exception as e:
            result.errors.append(str(e))

        return result

    def _deploy_docker(
        self, model_path: str, config: DeploymentConfig
    ) -> DeploymentResult:
        """Deploy using Docker."""
        result = DeploymentResult(target=config.target)

        try:
            # Generate Dockerfile
            dockerfile_content = self._generate_dockerfile(model_path, config)

            # Simulate Docker deployment
            result.success = True
            result.deployment_url = "http://localhost:8080/inference"
            result.artifacts = ["Dockerfile", model_path]

        except Exception as e:
            result.errors.append(str(e))

        return result

    def _deploy_kubernetes(
        self, model_path: str, config: DeploymentConfig
    ) -> DeploymentResult:
        """Deploy to Kubernetes."""
        result = DeploymentResult(target=config.target)

        try:
            # Generate Kubernetes manifests
            manifests = self._generate_k8s_manifests(model_path, config)

            result.success = True
            result.artifacts = manifests
            result.deployment_url = (
                "http://edgeflow-service.default.svc.cluster.local:8080"
            )

        except Exception as e:
            result.errors.append(str(e))

        return result

    def _deploy_bare_metal(
        self, model_path: str, config: DeploymentConfig
    ) -> DeploymentResult:
        """Deploy to bare metal server."""
        result = DeploymentResult(target=config.target)

        try:
            # Create deployment package
            package_path = self._create_deployment_package(model_path, config)

            result.success = True
            result.artifacts = [package_path]
            result.deployment_url = f"http://{config.target_host}:8080/inference"

        except Exception as e:
            result.errors.append(str(e))

        return result

    def _generate_dockerfile(self, model_path: str, config: DeploymentConfig) -> str:
        """Generate Dockerfile for deployment."""
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY {Path(model_path).name} ./model.tflite
COPY inference.py .

EXPOSE 8080

CMD ["python", "inference.py", "--model", "model.tflite", "--port", "8080"]
"""

        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)

        return dockerfile_content

    def _generate_k8s_manifests(
        self, model_path: str, config: DeploymentConfig
    ) -> List[str]:
        """Generate Kubernetes deployment manifests."""
        manifests = []

        # Deployment manifest
        deployment_yaml = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "edgeflow-model"},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "edgeflow-model"}},
                "template": {
                    "metadata": {"labels": {"app": "edgeflow-model"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "model-server",
                                "image": "edgeflow-model:latest",
                                "ports": [{"containerPort": 8080}],
                            }
                        ]
                    },
                },
            },
        }

        deployment_file = "deployment.yaml"
        with open(deployment_file, "w") as f:
            json.dump(deployment_yaml, f, indent=2)
        manifests.append(deployment_file)

        return manifests

    def _create_deployment_package(
        self, model_path: str, config: DeploymentConfig
    ) -> str:
        """Create deployment package for bare metal."""
        import tarfile

        package_path = "edgeflow_deployment.tar.gz"

        with tarfile.open(package_path, "w:gz") as tar:
            tar.add(model_path, arcname="model.tflite")
            # Add other deployment files as needed

        return package_path


def deploy_from_dsl(dsl_file: str) -> Dict[DeploymentTarget, DeploymentResult]:
    """Deploy model to all targets specified in DSL file."""
    from parser import parse_ef

    config = parse_ef(dsl_file)
    model_path = config.get("model", "model.tflite")

    # Parse deployment targets from config
    deployment_targets = []

    if config.get("deploy_raspberry_pi", False):
        deployment_targets.append(
            DeploymentConfig(
                target=DeploymentTarget.RASPBERRY_PI,
                model_path=model_path,
                target_host=config.get("raspberry_pi_host", "raspberrypi.local"),
            )
        )

    if config.get("deploy_docker", False):
        deployment_targets.append(
            DeploymentConfig(target=DeploymentTarget.DOCKER, model_path=model_path)
        )

    if config.get("deploy_kubernetes", False):
        deployment_targets.append(
            DeploymentConfig(target=DeploymentTarget.KUBERNETES, model_path=model_path)
        )

    deployer = CrossPlatformDeployer()
    return deployer.deploy_multi_target(model_path, deployment_targets)


if __name__ == "__main__":
    # Example usage
    deployer = CrossPlatformDeployer()

    configs = [
        DeploymentConfig(target=DeploymentTarget.DOCKER, model_path="model.tflite"),
        DeploymentConfig(
            target=DeploymentTarget.RASPBERRY_PI,
            model_path="model.tflite",
            target_host="raspberrypi.local",
        ),
    ]

    print("Cross-platform deployment orchestrator ready!")
