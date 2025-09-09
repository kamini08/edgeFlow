"""EdgeFlow Semantic Validator

This module validates the semantic correctness of EdgeFlow DSL configurations,
including model existence, parameter validation, and device compatibility.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


class EdgeFlowValidator:
    """Validates EdgeFlow DSL configurations for semantic correctness."""

    def __init__(self):
        self.supported_devices = {
            "raspberry_pi",
            "jetson_nano",
            "jetson_xavier",
            "cpu",
            "gpu",
        }
        self.supported_quantization = {"int8", "float16", "none", "off"}
        self.supported_input_streams = {"camera", "file", "stream", "sensor"}
        self.supported_optimization_goals = {"latency", "memory", "size", "balanced"}
        self.supported_fusion_options = {True, False, "true", "false"}

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a complete EdgeFlow configuration.

        Args:
            config: Parsed EdgeFlow configuration dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []

        try:
            # Validate required fields
            self._validate_required_fields(config, errors)

            # Validate model file
            self._validate_model_file(config, errors)

            # Validate quantization parameters
            self._validate_quantization(config, errors)

            # Validate device compatibility
            self._validate_device_compatibility(config, errors)

            # Validate input stream parameters
            self._validate_input_stream(config, errors)

            # Validate optimization parameters
            self._validate_optimization_params(config, errors)

            # Validate memory and buffer constraints
            self._validate_memory_constraints(config, errors)

            # Validate deployment path
            self._validate_deployment_path(config, errors)

            # Cross-parameter validation
            self._validate_cross_parameters(config, errors)

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return len(errors) == 0, errors

    def _validate_required_fields(
        self, config: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate that all required fields are present."""
        required_fields = ["model"]

        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

    def _validate_model_file(self, config: Dict[str, Any], errors: List[str]) -> None:
        """Validate that the model file exists and is accessible."""
        model_path = config.get("model")
        if not model_path:
            return

        # Check if file exists
        if not os.path.exists(model_path):
            errors.append(f"Model file not found: {model_path}")
            return

        # Check if it's a valid TensorFlow Lite file
        if not str(model_path).endswith((".tflite", ".lite")):
            errors.append(
                f"Model file must be a TensorFlow Lite file (.tflite): {model_path}"
            )
            return

        # Check file size (should be reasonable)
        try:
            file_size = os.path.getsize(model_path)
            if file_size < 1024:  # Less than 1KB
                errors.append(f"Model file too small (likely corrupted): {model_path}")
            elif file_size > 100 * 1024 * 1024:  # More than 100MB
                errors.append(
                    f"Model file too large for edge deployment: {model_path} ({file_size / 1024 / 1024:.1f}MB)"
                )
        except OSError as e:
            errors.append(f"Cannot access model file: {model_path} - {str(e)}")

    def _validate_quantization(self, config: Dict[str, Any], errors: List[str]) -> None:
        """Validate quantization parameters."""
        quantize = config.get("quantize", "none")

        if quantize not in self.supported_quantization:
            errors.append(
                f"Unsupported quantization type: {quantize}. Supported: {', '.join(self.supported_quantization)}"
            )

        # Check if quantization is compatible with model
        model_path = config.get("model")
        if quantize in ("int8", "float16") and model_path:
            if not str(model_path).endswith(".tflite"):
                errors.append(
                    f"Quantization requires TensorFlow Lite model (.tflite), got: {model_path}"
                )

    def _validate_device_compatibility(
        self, config: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate device compatibility."""
        device = config.get("target_device", "cpu")

        if device not in self.supported_devices:
            errors.append(
                f"Unsupported target device: {device}. Supported: {', '.join(self.supported_devices)}"
            )

        # Check device-specific constraints
        if device == "raspberry_pi":
            memory_limit = config.get("memory_limit", 64)
            try:
                memory_limit_num = float(memory_limit)
                if (
                    memory_limit_num > 512
                ):  # Pi typically has 1-8GB, but we're conservative
                    errors.append(
                        f"Memory limit too high for Raspberry Pi: {memory_limit}MB"
                    )
            except (ValueError, TypeError):
                pass  # Skip if not numeric

        elif device == "jetson_nano":
            memory_limit = config.get("memory_limit", 128)
            try:
                memory_limit_num = float(memory_limit)
                if memory_limit_num > 2048:  # Jetson Nano has 4GB
                    errors.append(
                        f"Memory limit too high for Jetson Nano: {memory_limit}MB"
                    )
            except (ValueError, TypeError):
                pass  # Skip if not numeric

    def _validate_input_stream(self, config: Dict[str, Any], errors: List[str]) -> None:
        """Validate input stream parameters."""
        input_stream = config.get("input_stream", "file")

        if input_stream not in self.supported_input_streams:
            errors.append(
                f"Unsupported input stream: {input_stream}. Supported: {', '.join(self.supported_input_streams)}"
            )

        # Validate buffer size for streaming
        if input_stream in ("camera", "stream", "sensor"):
            buffer_size = config.get("buffer_size", 1)
            try:
                buffer_size_num = int(buffer_size)
                if buffer_size_num < 1 or buffer_size_num > 128:
                    errors.append(
                        f"Buffer size must be between 1 and 128 for streaming input, got: {buffer_size}"
                    )
            except (ValueError, TypeError):
                pass  # Skip if not numeric

    def _validate_optimization_params(
        self, config: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate optimization parameters."""
        optimize_for = config.get("optimize_for", "balanced")

        if optimize_for not in self.supported_optimization_goals:
            errors.append(
                f"Unsupported optimization goal: {optimize_for}. Supported: {', '.join(self.supported_optimization_goals)}"
            )

        # Validate fusion setting
        fusion = config.get("enable_fusion", True)
        if fusion not in self.supported_fusion_options:
            errors.append(f"Invalid fusion setting: {fusion}. Must be true/false")

    def _validate_memory_constraints(
        self, config: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate memory and buffer constraints."""
        memory_limit = config.get("memory_limit")
        buffer_size = config.get("buffer_size")

        if memory_limit is not None:
            try:
                memory_limit_num = float(memory_limit)
                if memory_limit_num <= 0:
                    errors.append(
                        f"Memory limit must be a positive number, got: {memory_limit}"
                    )
                elif memory_limit_num < 16:  # Minimum 16MB
                    errors.append(
                        f"Memory limit too low (minimum 16MB), got: {memory_limit}MB"
                    )
            except (ValueError, TypeError):
                errors.append(f"Memory limit must be a number, got: {memory_limit}")

        if buffer_size is not None:
            try:
                buffer_size_num = int(buffer_size)
                if buffer_size_num <= 0:
                    errors.append(
                        f"Buffer size must be a positive integer, got: {buffer_size}"
                    )
                elif buffer_size_num > 256:  # Maximum 256
                    errors.append(
                        f"Buffer size too large (maximum 256), got: {buffer_size}"
                    )
            except (ValueError, TypeError):
                errors.append(f"Buffer size must be an integer, got: {buffer_size}")

    def _validate_deployment_path(
        self, config: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate deployment path."""
        deploy_path = config.get("deploy_path")
        if not deploy_path:
            return

        # Check if path is valid
        try:
            path = Path(deploy_path)
            if not path.is_absolute():
                errors.append(f"Deployment path must be absolute: {deploy_path}")

            # Check if parent directory exists (we don't create it automatically)
            if not path.parent.exists():
                errors.append(
                    f"Deployment path parent directory does not exist: {path.parent}"
                )
        except Exception as e:
            errors.append(f"Invalid deployment path: {deploy_path} - {str(e)}")

    def _validate_cross_parameters(
        self, config: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate cross-parameter dependencies."""
        quantize = config.get("quantize", "none")
        optimize_for = config.get("optimize_for", "balanced")
        device = config.get("target_device", "cpu")
        memory_limit = config.get("memory_limit")

        # Check quantization vs optimization goal compatibility
        if quantize == "int8" and optimize_for == "memory":
            # INT8 quantization is good for memory optimization
            pass
        elif quantize == "float16" and optimize_for == "latency":
            # FLOAT16 might not be optimal for latency on some devices
            if device == "raspberry_pi":
                errors.append(
                    "FLOAT16 quantization may not improve latency on Raspberry Pi"
                )

        # Check memory limit vs device capabilities
        if memory_limit and device == "raspberry_pi":
            try:
                memory_limit_num = float(memory_limit)
                if memory_limit_num > 256:
                    errors.append("Memory limit too high for Raspberry Pi deployment")
            except (ValueError, TypeError):
                pass  # Skip if not numeric

        # Check buffer size vs memory limit
        buffer_size = config.get("buffer_size", 1)
        if memory_limit and buffer_size:
            try:
                memory_limit_num = float(memory_limit)
                buffer_size_num = int(buffer_size)
                # More realistic buffer memory estimation (224x224x3 float32 = ~0.6MB per buffer)
                estimated_buffer_memory = buffer_size_num * 0.6  # MB per buffer
                if (
                    estimated_buffer_memory > memory_limit_num * 0.8
                ):  # Buffer shouldn't use more than 80% of memory
                    errors.append(
                        f"Buffer size too large for memory limit: {buffer_size} buffers (~{estimated_buffer_memory:.1f}MB) with {memory_limit}MB limit"
                    )
            except (ValueError, TypeError):
                pass  # Skip this check if values are not numeric

    def validate_model_compatibility(
        self, model_path: str, config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate that the model is compatible with the configuration.

        Args:
            model_path: Path to the model file
            config: EdgeFlow configuration

        Returns:
            Tuple of (is_compatible, list_of_warnings)
        """
        warnings = []

        try:
            # Check model size vs device constraints
            if os.path.exists(model_path):
                model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                device = config.get("target_device", "cpu")
                memory_limit = config.get("memory_limit", 64)

                if device == "raspberry_pi" and model_size_mb > 50:
                    warnings.append(
                        f"Large model ({model_size_mb:.1f}MB) may be slow on Raspberry Pi"
                    )

                try:
                    memory_limit_num = float(memory_limit)
                    if model_size_mb > memory_limit_num * 0.8:
                        warnings.append(
                            f"Model size ({model_size_mb:.1f}MB) is close to memory limit ({memory_limit}MB)"
                        )
                except (ValueError, TypeError):
                    pass  # Skip if memory_limit is not numeric

                # Check quantization compatibility
                quantize = config.get("quantize", "none")
                if quantize == "int8" and not str(model_path).endswith(".tflite"):
                    warnings.append(
                        "INT8 quantization works best with TensorFlow Lite models"
                    )

        except Exception as e:
            warnings.append(f"Could not validate model compatibility: {str(e)}")

        return len(warnings) == 0, warnings


def validate_edgeflow_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Main validation function for EdgeFlow configurations.

    Args:
        config: Parsed EdgeFlow configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = EdgeFlowValidator()
    return validator.validate_config(config)


def validate_model_compatibility(
    model_path: str, config: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Validate model compatibility with configuration.

    Args:
        model_path: Path to the model file
        config: EdgeFlow configuration

    Returns:
        Tuple of (is_compatible, list_of_warnings)
    """
    validator = EdgeFlowValidator()
    return validator.validate_model_compatibility(model_path, config)


if __name__ == "__main__":
    # Test the validator
    test_config = {
        "model": "mobilenet_v2.tflite",
        "quantize": "int8",
        "target_device": "raspberry_pi",
        "deploy_path": "/models/",
        "input_stream": "camera",
        "buffer_size": 32,
        "optimize_for": "latency",
        "memory_limit": 64,
        "enable_fusion": True,
    }

    is_valid, errors = validate_edgeflow_config(test_config)
    print(f"Configuration valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")

    is_compatible, warnings = validate_model_compatibility(
        "mobilenet_v2.tflite", test_config
    )
    print(f"Model compatible: {is_compatible}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
