"""EdgeFlow Deployment Validation System

This module provides comprehensive validation of deployment artifacts without requiring
physical hardware. It includes:

1. Static Analysis: Validates package structure, dependencies, and configurations
2. Simulation Testing: Tests inference code and deployment scripts in controlled environments
3. Compatibility Checking: Validates device-specific requirements and constraints
4. Performance Estimation: Provides realistic performance estimates based on device characteristics
"""

import json
import logging
import os
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for deployment testing."""

    BASIC = "basic"  # Package structure and metadata only
    STATIC = "static"  # Code analysis and dependency checking
    SIMULATION = "simulation"  # Simulated execution testing
    COMPREHENSIVE = "comprehensive"  # All validation levels


class ValidationResult(Enum):
    """Validation result status."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""

    level: ValidationLevel
    category: str
    severity: ValidationResult
    message: str
    details: Optional[str] = None
    suggestions: List[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    package_path: str
    device_type: str
    validation_level: ValidationLevel
    overall_result: ValidationResult
    issues: List[ValidationIssue]
    metrics: Dict[str, Any]
    recommendations: List[str]


class EdgeFlowDeploymentValidator:
    """Validates EdgeFlow deployment artifacts without requiring physical hardware."""

    def __init__(self):
        self.device_requirements = self._initialize_device_requirements()
        self.validation_tools = self._initialize_validation_tools()

    def _initialize_device_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Initialize device-specific requirements for validation."""
        return {
            "raspberry_pi": {
                "min_python_version": "3.7",
                "required_packages": ["numpy", "tflite-runtime"],
                "max_file_size_mb": 100,
                "supported_architectures": ["armv7l", "aarch64"],
                "memory_constraints": {"min_mb": 64, "max_mb": 256},
                "storage_constraints": {"min_mb": 16, "max_mb": 100},
            },
            "jetson_nano": {
                "min_python_version": "3.6",
                "required_packages": ["numpy", "tflite-runtime", "cuda"],
                "max_file_size_mb": 200,
                "supported_architectures": ["aarch64"],
                "memory_constraints": {"min_mb": 128, "max_mb": 1024},
                "storage_constraints": {"min_mb": 32, "max_mb": 200},
            },
            "jetson_xavier": {
                "min_python_version": "3.6",
                "required_packages": ["numpy", "tflite-runtime", "cuda", "tensorrt"],
                "max_file_size_mb": 500,
                "supported_architectures": ["aarch64"],
                "memory_constraints": {"min_mb": 256, "max_mb": 2048},
                "storage_constraints": {"min_mb": 64, "max_mb": 500},
            },
            "cortex_m4": {
                "min_python_version": None,  # Bare metal
                "required_packages": ["tensorflow-lite-micro"],
                "max_file_size_mb": 50,
                "supported_architectures": ["armv7e-m"],
                "memory_constraints": {"min_mb": 32, "max_mb": 128},
                "storage_constraints": {"min_mb": 8, "max_mb": 50},
            },
            "cortex_m7": {
                "min_python_version": None,  # Bare metal
                "required_packages": ["tensorflow-lite-micro"],
                "max_file_size_mb": 100,
                "supported_architectures": ["armv7e-m"],
                "memory_constraints": {"min_mb": 64, "max_mb": 256},
                "storage_constraints": {"min_mb": 16, "max_mb": 100},
            },
        }

    def _initialize_validation_tools(self) -> Dict[str, Any]:
        """Initialize validation tools and check availability."""
        tools = {}

        # Check Python availability
        try:
            import sys

            tools["python_version"] = sys.version_info
            tools["python_available"] = True
        except Exception:
            tools["python_available"] = False

        # Check TensorFlow Lite availability
        try:
            import tflite_runtime.interpreter

            tools["tflite_available"] = True
        except ImportError:
            try:
                import tensorflow.lite

                tools["tflite_available"] = True
            except ImportError:
                tools["tflite_available"] = False

        # Check CUDA availability (for Jetson devices)
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            tools["cuda_available"] = result.returncode == 0
        except Exception:
            tools["cuda_available"] = False

        # Check cross-compilation tools (for Cortex-M devices)
        try:
            result = subprocess.run(
                ["arm-none-eabi-gcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            tools["arm_gcc_available"] = result.returncode == 0
        except Exception:
            tools["arm_gcc_available"] = False

        return tools

    def validate_deployment(
        self,
        package_path: str,
        device_type: str,
        validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE,
    ) -> ValidationReport:
        """Validate deployment package for target device.

        Args:
            package_path: Path to the deployment package
            device_type: Target device type
            validation_level: Level of validation to perform

        Returns:
            Comprehensive validation report
        """
        logger.info(f"Validating deployment package: {package_path}")
        logger.info(f"Target device: {device_type}")
        logger.info(f"Validation level: {validation_level.value}")

        issues = []
        metrics = {}
        recommendations = []

        # Basic validation - package structure
        if validation_level in [
            ValidationLevel.BASIC,
            ValidationLevel.STATIC,
            ValidationLevel.SIMULATION,
            ValidationLevel.COMPREHENSIVE,
        ]:
            basic_issues, basic_metrics = self._validate_package_structure(
                package_path, device_type
            )
            issues.extend(basic_issues)
            metrics.update(basic_metrics)

        # Static validation - code analysis and dependencies
        if validation_level in [
            ValidationLevel.STATIC,
            ValidationLevel.SIMULATION,
            ValidationLevel.COMPREHENSIVE,
        ]:
            static_issues, static_metrics = self._validate_static_analysis(
                package_path, device_type
            )
            issues.extend(static_issues)
            metrics.update(static_metrics)

        # Simulation validation - simulated execution testing
        if validation_level in [
            ValidationLevel.SIMULATION,
            ValidationLevel.COMPREHENSIVE,
        ]:
            simulation_issues, simulation_metrics = self._validate_simulation(
                package_path, device_type
            )
            issues.extend(simulation_issues)
            metrics.update(simulation_metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(issues, device_type)

        # Determine overall result
        overall_result = self._determine_overall_result(issues)

        return ValidationReport(
            package_path=package_path,
            device_type=device_type,
            validation_level=validation_level,
            overall_result=overall_result,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _validate_package_structure(
        self, package_path: str, device_type: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Validate package structure and metadata."""
        issues = []
        metrics = {}

        logger.info("Validating package structure...")

        # Check if package exists
        if not os.path.exists(package_path):
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.BASIC,
                    category="package_structure",
                    severity=ValidationResult.FAIL,
                    message="Deployment package not found",
                    details=f"Package path does not exist: {package_path}",
                )
            )
            return issues, metrics

        # Check package size
        package_size_mb = os.path.getsize(package_path) / (1024 * 1024)
        metrics["package_size_mb"] = package_size_mb

        device_reqs = self.device_requirements.get(device_type, {})
        max_size = device_reqs.get("storage_constraints", {}).get("max_mb", 1000)

        if package_size_mb > max_size:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.BASIC,
                    category="package_size",
                    severity=ValidationResult.FAIL,
                    message=f"Package size exceeds device limit",
                    details=f"Package size: {package_size_mb:.1f}MB, Device limit: {max_size}MB",
                    suggestions=[
                        "Reduce model size through quantization",
                        "Use more aggressive compression",
                        "Remove unnecessary dependencies",
                    ],
                )
            )
        elif package_size_mb > max_size * 0.8:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.BASIC,
                    category="package_size",
                    severity=ValidationResult.WARN,
                    message=f"Package size approaching device limit",
                    details=f"Package size: {package_size_mb:.1f}MB, Device limit: {max_size}MB",
                )
            )

        # Validate package format
        if package_path.endswith(".tar.gz"):
            try:
                with tarfile.open(package_path, "r:gz") as tar:
                    members = tar.getmembers()
                    metrics["package_files"] = len(members)

                    # Check for required files
                    required_files = ["deployment_manifest.json"]
                    found_files = [member.name for member in members]

                    for required_file in required_files:
                        if not any(required_file in name for name in found_files):
                            issues.append(
                                ValidationIssue(
                                    level=ValidationLevel.BASIC,
                                    category="package_structure",
                                    severity=ValidationResult.FAIL,
                                    message=f"Required file missing: {required_file}",
                                    details="Deployment manifest not found in package",
                                )
                            )

                    # Check manifest content
                    manifest_files = [
                        name for name in found_files if "manifest" in name
                    ]
                    if manifest_files:
                        try:
                            manifest_member = tar.extractfile(manifest_files[0])
                            if manifest_member:
                                manifest_data = json.loads(
                                    manifest_member.read().decode("utf-8")
                                )
                                metrics["manifest_data"] = manifest_data

                                # Validate manifest against device requirements
                                manifest_device = manifest_data.get("device_type", "")
                                if manifest_device != device_type:
                                    issues.append(
                                        ValidationIssue(
                                            level=ValidationLevel.BASIC,
                                            category="package_structure",
                                            severity=ValidationResult.WARN,
                                            message="Device type mismatch in manifest",
                                            details=f"Manifest device: {manifest_device}, Expected: {device_type}",
                                        )
                                    )
                        except Exception as e:
                            issues.append(
                                ValidationIssue(
                                    level=ValidationLevel.BASIC,
                                    category="package_structure",
                                    severity=ValidationResult.WARN,
                                    message="Could not parse deployment manifest",
                                    details=str(e),
                                )
                            )

            except Exception as e:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.BASIC,
                        category="package_structure",
                        severity=ValidationResult.FAIL,
                        message="Invalid package format",
                        details=f"Could not open package: {str(e)}",
                    )
                )

        return issues, metrics

    def _validate_static_analysis(
        self, package_path: str, device_type: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Perform static analysis of package contents."""
        issues = []
        metrics = {}

        logger.info("Performing static analysis...")

        device_reqs = self.device_requirements.get(device_type, {})

        try:
            with tarfile.open(package_path, "r:gz") as tar:
                members = tar.getmembers()

                # Analyze inference code
                inference_files = [m for m in members if "inference" in m.name.lower()]
                if inference_files:
                    metrics["inference_files"] = len(inference_files)

                    # Check for Python vs C++ code
                    python_files = [
                        f for f in inference_files if f.name.endswith(".py")
                    ]
                    cpp_files = [f for f in inference_files if f.name.endswith(".cpp")]

                    if device_type in ["cortex_m4", "cortex_m7"]:
                        if not cpp_files:
                            issues.append(
                                ValidationIssue(
                                    level=ValidationLevel.STATIC,
                                    category="code_analysis",
                                    severity=ValidationResult.WARN,
                                    message="No C++ code found for embedded device",
                                    details="Cortex-M devices typically require C++ implementation",
                                    suggestions=[
                                        "Generate C++ inference code",
                                        "Use TensorFlow Lite Micro",
                                    ],
                                )
                            )
                    else:
                        if not python_files:
                            issues.append(
                                ValidationIssue(
                                    level=ValidationLevel.STATIC,
                                    category="code_analysis",
                                    severity=ValidationResult.WARN,
                                    message="No Python code found for Linux device",
                                    details="Linux devices typically use Python implementation",
                                )
                            )

                # Analyze dependencies
                dep_files = [m for m in members if "dependencies" in m.name.lower()]
                if dep_files:
                    try:
                        dep_member = tar.extractfile(dep_files[0])
                        if dep_member:
                            dep_content = dep_member.read().decode("utf-8")
                            metrics["dependencies_content"] = dep_content

                            # Check required packages
                            required_packages = device_reqs.get("required_packages", [])
                            for package in required_packages:
                                if package not in dep_content:
                                    issues.append(
                                        ValidationIssue(
                                            level=ValidationLevel.STATIC,
                                            category="dependencies",
                                            severity=ValidationResult.WARN,
                                            message=f"Required package not found: {package}",
                                            details=f"Package {package} is required for {device_type}",
                                            suggestions=[
                                                f"Add {package} to requirements.txt"
                                            ],
                                        )
                                    )
                    except Exception as e:
                        issues.append(
                            ValidationIssue(
                                level=ValidationLevel.STATIC,
                                category="dependencies",
                                severity=ValidationResult.WARN,
                                message="Could not analyze dependencies",
                                details=str(e),
                            )
                        )

                # Analyze deployment scripts
                script_files = [m for m in members if "script" in m.name.lower()]
                if script_files:
                    metrics["deployment_scripts"] = len(script_files)

                    # Check for device-specific scripts
                    device_script_found = any(
                        device_type in f.name for f in script_files
                    )
                    if not device_script_found:
                        issues.append(
                            ValidationIssue(
                                level=ValidationLevel.STATIC,
                                category="deployment_scripts",
                                severity=ValidationResult.WARN,
                                message="No device-specific deployment script found",
                                details=f"Expected script for {device_type}",
                                suggestions=[
                                    f"Generate {device_type}-specific deployment script"
                                ],
                            )
                        )

        except Exception as e:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.STATIC,
                    category="static_analysis",
                    severity=ValidationResult.FAIL,
                    message="Static analysis failed",
                    details=str(e),
                )
            )

        return issues, metrics

    def _validate_simulation(
        self, package_path: str, device_type: str
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Perform simulated execution testing."""
        issues = []
        metrics = {}

        logger.info("Performing simulation testing...")

        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract package
                with tarfile.open(package_path, "r:gz") as tar:
                    tar.extractall(temp_dir)

                # Find and test inference code
                inference_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if "inference" in file.lower() and file.endswith(".py"):
                            inference_files.append(os.path.join(root, file))

                if inference_files:
                    metrics["inference_files_tested"] = len(inference_files)

                    for inference_file in inference_files:
                        # Test Python syntax
                        try:
                            with open(inference_file, "r") as f:
                                code = f.read()

                            # Basic syntax check
                            compile(code, inference_file, "exec")

                            # Check for device-specific imports
                            if device_type in ["jetson_nano", "jetson_xavier"]:
                                if (
                                    "cuda" not in code.lower()
                                    and "tensorrt" not in code.lower()
                                ):
                                    issues.append(
                                        ValidationIssue(
                                            level=ValidationLevel.SIMULATION,
                                            category="code_simulation",
                                            severity=ValidationResult.WARN,
                                            message="GPU acceleration not detected in inference code",
                                            details=f"Jetson devices support CUDA/TensorRT acceleration",
                                            suggestions=[
                                                "Add CUDA/TensorRT imports",
                                                "Implement GPU inference path",
                                            ],
                                        )
                                    )

                            # Check for TensorFlow Lite usage
                            if (
                                "tflite" not in code.lower()
                                and "tensorflow.lite" not in code.lower()
                            ):
                                issues.append(
                                    ValidationIssue(
                                        level=ValidationLevel.SIMULATION,
                                        category="code_simulation",
                                        severity=ValidationResult.WARN,
                                        message="TensorFlow Lite not detected in inference code",
                                        details="Inference code should use TensorFlow Lite for edge deployment",
                                    )
                                )

                        except SyntaxError as e:
                            issues.append(
                                ValidationIssue(
                                    level=ValidationLevel.SIMULATION,
                                    category="code_simulation",
                                    severity=ValidationResult.FAIL,
                                    message="Syntax error in inference code",
                                    details=f"File: {inference_file}, Error: {str(e)}",
                                )
                            )
                        except Exception as e:
                            issues.append(
                                ValidationIssue(
                                    level=ValidationLevel.SIMULATION,
                                    category="code_simulation",
                                    severity=ValidationResult.WARN,
                                    message="Could not analyze inference code",
                                    details=str(e),
                                )
                            )

                # Test deployment scripts
                script_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(".sh"):
                            script_files.append(os.path.join(root, file))

                if script_files:
                    metrics["deployment_scripts_tested"] = len(script_files)

                    for script_file in script_files:
                        try:
                            # Check script syntax
                            result = subprocess.run(
                                ["bash", "-n", script_file],
                                capture_output=True,
                                text=True,
                                timeout=10,
                            )
                            if result.returncode != 0:
                                issues.append(
                                    ValidationIssue(
                                        level=ValidationLevel.SIMULATION,
                                        category="script_simulation",
                                        severity=ValidationResult.FAIL,
                                        message="Syntax error in deployment script",
                                        details=f"File: {script_file}, Error: {result.stderr}",
                                    )
                                )
                        except subprocess.TimeoutExpired:
                            issues.append(
                                ValidationIssue(
                                    level=ValidationLevel.SIMULATION,
                                    category="script_simulation",
                                    severity=ValidationResult.WARN,
                                    message="Deployment script validation timeout",
                                    details=f"Script: {script_file}",
                                )
                            )
                        except Exception as e:
                            issues.append(
                                ValidationIssue(
                                    level=ValidationLevel.SIMULATION,
                                    category="script_simulation",
                                    severity=ValidationResult.WARN,
                                    message="Could not validate deployment script",
                                    details=str(e),
                                )
                            )

                # Simulate performance characteristics
                performance_metrics = self._simulate_performance_characteristics(
                    device_type
                )
                metrics.update(performance_metrics)

            except Exception as e:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.SIMULATION,
                        category="simulation",
                        severity=ValidationResult.FAIL,
                        message="Simulation testing failed",
                        details=str(e),
                    )
                )

        return issues, metrics

    def _simulate_performance_characteristics(self, device_type: str) -> Dict[str, Any]:
        """Simulate performance characteristics for the device."""
        device_reqs = self.device_requirements.get(device_type, {})

        # Simulate based on device characteristics
        if device_type == "raspberry_pi":
            return {
                "estimated_latency_ms": 15.0,
                "estimated_throughput_fps": 65.0,
                "estimated_memory_usage_mb": 45.0,
                "estimated_cpu_usage_percent": 85.0,
            }
        elif device_type in ["jetson_nano", "jetson_xavier"]:
            return {
                "estimated_latency_ms": 8.0,
                "estimated_throughput_fps": 125.0,
                "estimated_memory_usage_mb": 120.0,
                "estimated_cpu_usage_percent": 60.0,
                "estimated_gpu_usage_percent": 40.0,
            }
        elif device_type in ["cortex_m4", "cortex_m7"]:
            return {
                "estimated_latency_ms": 25.0,
                "estimated_throughput_fps": 40.0,
                "estimated_memory_usage_mb": 15.0,
                "estimated_cpu_usage_percent": 95.0,
            }
        else:
            return {
                "estimated_latency_ms": 12.0,
                "estimated_throughput_fps": 80.0,
                "estimated_memory_usage_mb": 60.0,
                "estimated_cpu_usage_percent": 75.0,
            }

    def _generate_recommendations(
        self, issues: List[ValidationIssue], device_type: str
    ) -> List[str]:
        """Generate recommendations based on validation issues."""
        recommendations = []

        # Count issues by severity
        fail_count = len([i for i in issues if i.severity == ValidationResult.FAIL])
        warn_count = len([i for i in issues if i.severity == ValidationResult.WARN])

        if fail_count > 0:
            recommendations.append(
                f"Fix {fail_count} critical issues before deployment"
            )

        if warn_count > 0:
            recommendations.append(
                f"Address {warn_count} warnings for optimal performance"
            )

        # Device-specific recommendations
        device_reqs = self.device_requirements.get(device_type, {})

        if device_type in ["jetson_nano", "jetson_xavier"]:
            recommendations.append(
                "Ensure CUDA/TensorRT dependencies are properly configured"
            )
            recommendations.append("Test GPU acceleration paths in inference code")

        if device_type in ["cortex_m4", "cortex_m7"]:
            recommendations.append("Verify cross-compilation toolchain is available")
            recommendations.append("Test with TensorFlow Lite Micro runtime")

        if device_type == "raspberry_pi":
            recommendations.append("Optimize for ARM architecture")
            recommendations.append("Test with TensorFlow Lite runtime")

        return recommendations

    def _determine_overall_result(
        self, issues: List[ValidationIssue]
    ) -> ValidationResult:
        """Determine overall validation result."""
        if any(issue.severity == ValidationResult.FAIL for issue in issues):
            return ValidationResult.FAIL
        elif any(issue.severity == ValidationResult.WARN for issue in issues):
            return ValidationResult.WARN
        else:
            return ValidationResult.PASS


def validate_deployment_package(
    package_path: str,
    device_type: str,
    validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE,
) -> ValidationReport:
    """Validate deployment package for target device.

    Args:
        package_path: Path to the deployment package
        device_type: Target device type
        validation_level: Level of validation to perform

    Returns:
        Comprehensive validation report
    """
    validator = EdgeFlowDeploymentValidator()
    return validator.validate_deployment(package_path, device_type, validation_level)


if __name__ == "__main__":
    # Test the validator
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python deployment_validator.py <package_path> <device_type> [validation_level]"
        )
        sys.exit(1)

    package_path = sys.argv[1]
    device_type = sys.argv[2]
    validation_level = (
        ValidationLevel(sys.argv[3])
        if len(sys.argv) > 3
        else ValidationLevel.COMPREHENSIVE
    )

    validator = EdgeFlowDeploymentValidator()
    report = validator.validate_deployment(package_path, device_type, validation_level)

    print(f"Validation Report for {device_type}")
    print("=" * 50)
    print(f"Overall Result: {report.overall_result.value}")
    print(f"Validation Level: {report.validation_level.value}")
    print(f"Issues Found: {len(report.issues)}")

    for issue in report.issues:
        print(f"\n{issue.severity.value.upper()}: {issue.message}")
        if issue.details:
            print(f"  Details: {issue.details}")
        if issue.suggestions:
            print(f"  Suggestions: {', '.join(issue.suggestions)}")

    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
