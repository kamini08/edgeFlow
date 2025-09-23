"""EdgeFlow Static Validation Module

This module provides comprehensive compile-time validation for EdgeFlow DSL configurations,
identifying incompatible combinations, providing clear error messages, and suggesting
corrected configurations or default fallbacks.

Key Features:
- Early validation to catch errors before expensive operations
- Clear, actionable error messages with suggestions
- Configuration compatibility matrix
- Automatic fallback suggestions
- Performance impact estimation
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationCategory(Enum):
    """Categories of validation issues."""

    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class ValidationIssue:
    """Represents a validation issue with context and suggestions."""

    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    parameter: Optional[str] = None
    current_value: Optional[Any] = None
    suggested_value: Optional[Any] = None
    explanation: Optional[str] = None
    impact: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    suggestions: List[ValidationIssue] = field(default_factory=list)
    estimated_performance_impact: Dict[str, Any] = field(default_factory=dict)
    compatibility_score: float = 1.0


class EdgeFlowStaticValidator:
    """Comprehensive static validator for EdgeFlow configurations."""

    def __init__(self):
        # Supported values for each parameter
        self.supported_values = {
            "quantize": {"int8", "float16", "none", "off"},
            "target_device": {
                "raspberry_pi",
                "jetson_nano",
                "jetson_xavier",
                "cortex_m4",
                "cortex_m7",
                "cpu",
                "gpu",
                "tpu",
                "npu",
            },
            "input_stream": {"camera", "file", "stream", "sensor", "microphone"},
            "optimize_for": {
                "latency",
                "memory",
                "size",
                "balanced",
                "accuracy",
                "power",
            },
            "enable_fusion": {True, False, "true", "false"},
            "enable_pruning": {True, False, "true", "false"},
        }

        # Device capabilities and constraints
        self.device_capabilities = {
            "raspberry_pi": {
                "max_memory_mb": 2048,
                "max_model_size_mb": 100,
                "supports_fp16": False,
                "supports_int8": True,
                "supports_fusion": True,
                "supports_pruning": True,
                "recommended_buffer_size": 16,
                "performance_tier": "low",
            },
            "jetson_nano": {
                "max_memory_mb": 4096,
                "max_model_size_mb": 200,
                "supports_fp16": True,
                "supports_int8": True,
                "supports_fusion": True,
                "supports_pruning": True,
                "recommended_buffer_size": 32,
                "performance_tier": "medium",
            },
            "jetson_xavier": {
                "max_memory_mb": 8192,
                "max_model_size_mb": 500,
                "supports_fp16": True,
                "supports_int8": True,
                "supports_fusion": True,
                "supports_pruning": True,
                "recommended_buffer_size": 64,
                "performance_tier": "high",
            },
            "cortex_m4": {
                "max_memory_mb": 512,
                "max_model_size_mb": 50,
                "supports_fp16": False,
                "supports_int8": True,
                "supports_fusion": False,
                "supports_pruning": True,
                "recommended_buffer_size": 4,
                "performance_tier": "ultra_low",
            },
            "cortex_m7": {
                "max_memory_mb": 1024,
                "max_model_size_mb": 100,
                "supports_fp16": False,
                "supports_int8": True,
                "supports_fusion": True,
                "supports_pruning": True,
                "recommended_buffer_size": 8,
                "performance_tier": "low",
            },
            "cpu": {
                "max_memory_mb": 8192,
                "max_model_size_mb": 1000,
                "supports_fp16": True,
                "supports_int8": True,
                "supports_fusion": True,
                "supports_pruning": True,
                "recommended_buffer_size": 64,
                "performance_tier": "medium",
            },
            "gpu": {
                "max_memory_mb": 16384,
                "max_model_size_mb": 2000,
                "supports_fp16": True,
                "supports_int8": True,
                "supports_fusion": True,
                "supports_pruning": True,
                "recommended_buffer_size": 128,
                "performance_tier": "high",
            },
            "tpu": {
                "max_memory_mb": 32768,
                "max_model_size_mb": 5000,
                "supports_fp16": True,
                "supports_int8": True,
                "supports_fusion": True,
                "supports_pruning": True,
                "recommended_buffer_size": 256,
                "performance_tier": "ultra_high",
            },
            "npu": {
                "max_memory_mb": 8192,
                "max_model_size_mb": 1000,
                "supports_fp16": True,
                "supports_int8": True,
                "supports_fusion": True,
                "supports_pruning": True,
                "recommended_buffer_size": 64,
                "performance_tier": "high",
            },
        }

        # Model format compatibility
        self.model_format_compatibility = {
            ".tflite": {
                "quantize_int8": True,
                "quantize_float16": True,
                "fusion": True,
                "pruning": True,
                "recommended_for": [
                    "raspberry_pi",
                    "jetson_nano",
                    "cortex_m4",
                    "cortex_m7",
                ],
            },
            ".h5": {
                "quantize_int8": True,
                "quantize_float16": False,
                "fusion": True,
                "pruning": True,
                "recommended_for": ["cpu", "gpu"],
            },
            ".keras": {
                "quantize_int8": True,
                "quantize_float16": False,
                "fusion": True,
                "pruning": True,
                "recommended_for": ["cpu", "gpu"],
            },
            ".onnx": {
                "quantize_int8": True,
                "quantize_float16": True,
                "fusion": True,
                "pruning": True,
                "recommended_for": ["jetson_xavier", "gpu", "tpu"],
            },
            ".pb": {
                "quantize_int8": False,
                "quantize_float16": False,
                "fusion": True,
                "pruning": True,
                "recommended_for": ["cpu", "gpu"],
            },
            ".pth": {
                "quantize_int8": False,
                "quantize_float16": False,
                "fusion": False,
                "pruning": True,
                "recommended_for": ["cpu", "gpu"],
            },
        }

        # Incompatible combinations
        self.incompatible_combinations = [
            # Device + Quantization incompatibilities
            (
                "cortex_m4",
                "quantize",
                "float16",
                "Cortex-M4 doesn't support FP16 operations",
            ),
            (
                "cortex_m7",
                "quantize",
                "float16",
                "Cortex-M7 doesn't support FP16 operations",
            ),
            # Device + Fusion incompatibilities
            (
                "cortex_m4",
                "enable_fusion",
                True,
                "Cortex-M4 doesn't support operation fusion",
            ),
            # Model format + Quantization incompatibilities
            (
                ".pb",
                "quantize",
                "int8",
                "TensorFlow SavedModel doesn't support INT8 quantization",
            ),
            (
                ".pth",
                "quantize",
                "int8",
                "PyTorch models don't support INT8 quantization",
            ),
            (
                ".pb",
                "quantize",
                "float16",
                "TensorFlow SavedModel doesn't support FP16 quantization",
            ),
            (
                ".pth",
                "quantize",
                "float16",
                "PyTorch models don't support FP16 quantization",
            ),
            # Model format + Fusion incompatibilities
            (
                ".pth",
                "enable_fusion",
                True,
                "PyTorch models don't support operation fusion",
            ),
            # Optimization goal conflicts
            (
                "optimize_for",
                "accuracy",
                "quantize",
                "int8",
                "INT8 quantization reduces accuracy",
            ),
            (
                "optimize_for",
                "latency",
                "enable_pruning",
                True,
                "Pruning may increase latency",
            ),
            # Memory constraints
            (
                "buffer_size",
                ">",
                "memory_limit",
                "Buffer size too large for memory limit",
            ),
        ]

    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Comprehensive validation of EdgeFlow configuration.

        Args:
            config: Parsed EdgeFlow configuration dictionary

        Returns:
            ValidationResult with detailed validation information
        """
        result = ValidationResult(is_valid=True)

        try:
            # Basic parameter validation
            self._validate_basic_parameters(config, result)

            # Cross-parameter compatibility validation
            self._validate_cross_parameter_compatibility(config, result)

            # Device-specific validation
            self._validate_device_specific_constraints(config, result)

            # Model format compatibility
            self._validate_model_format_compatibility(config, result)

            # Performance impact estimation
            self._estimate_performance_impact(config, result)

            # Calculate compatibility score
            self._calculate_compatibility_score(result)

            # Determine overall validity
            result.is_valid = not any(
                issue.severity == ValidationSeverity.ERROR for issue in result.issues
            )

        except Exception as e:
            logger.error(f"Validation error: {e}")
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SYNTAX,
                    message=f"Validation failed: {str(e)}",
                )
            )
            result.is_valid = False

        return result

    def _validate_basic_parameters(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate basic parameter values and types."""

        # Required parameters
        if "model" not in config:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SYNTAX,
                    message="Missing required parameter: 'model'",
                    parameter="model",
                    suggested_value="path/to/your/model.tflite",
                    explanation="The model parameter specifies the path to your ML model file",
                )
            )

        # Validate parameter values
        for param, supported_values in self.supported_values.items():
            if param in config:
                value = config[param]
                if value not in supported_values:
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category=ValidationCategory.SYNTAX,
                            message=f"Invalid value for '{param}': {value}",
                            parameter=param,
                            current_value=value,
                            suggested_value=list(supported_values)[0],
                            explanation=f"Supported values: {', '.join(map(str, supported_values))}",
                        )
                    )

        # Validate numeric parameters
        numeric_params = {
            "buffer_size": (1, 256, "Buffer size must be between 1 and 256"),
            "memory_limit": (16, 32768, "Memory limit must be between 16MB and 32GB"),
            "pruning_sparsity": (
                0.0,
                1.0,
                "Pruning sparsity must be between 0.0 and 1.0",
            ),
        }

        for param, (min_val, max_val, error_msg) in numeric_params.items():
            if param in config:
                try:
                    value = float(config[param])
                    if not (min_val <= value <= max_val):
                        result.issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category=ValidationCategory.SYNTAX,
                                message=f"{error_msg}, got: {value}",
                                parameter=param,
                                current_value=value,
                                suggested_value=max(min_val, min(value, max_val)),
                            )
                        )
                except (ValueError, TypeError):
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category=ValidationCategory.SYNTAX,
                            message=f"'{param}' must be a number, got: {type(config[param])}",
                            parameter=param,
                            current_value=config[param],
                            suggested_value=min_val,
                        )
                    )

    def _validate_cross_parameter_compatibility(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate compatibility between different parameters."""

        # Check incompatible combinations
        for combination in self.incompatible_combinations:
            if len(combination) == 4:
                param1, param2, value, reason = combination
                if config.get(param1) == value and param2 in config:
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category=ValidationCategory.COMPATIBILITY,
                            message=f"Incompatible combination: {param1}={value} with {param2}",
                            parameter=param2,
                            explanation=reason,
                            impact="This combination will cause compilation or runtime errors",
                        )
                    )
            elif len(combination) == 5:
                param1, value1, param2, value2, reason = combination
                if config.get(param1) == value1 and config.get(param2) == value2:
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category=ValidationCategory.COMPATIBILITY,
                            message=f"Suboptimal combination: {param1}={value1} with {param2}={value2}",
                            parameter=param2,
                            explanation=reason,
                            impact="This combination may not achieve optimal performance",
                        )
                    )

        # Check optimization goal conflicts
        optimize_for = config.get("optimize_for", "balanced")
        quantize = config.get("quantize", "none")

        if optimize_for == "accuracy" and quantize == "int8":
            result.warnings.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.PERFORMANCE,
                    message="INT8 quantization reduces model accuracy",
                    parameter="quantize",
                    current_value="int8",
                    suggested_value="float16",
                    explanation="Consider float16 quantization for better accuracy",
                    impact="May reduce accuracy by 2-5%",
                )
            )

        if optimize_for == "latency" and config.get("enable_pruning", False):
            result.warnings.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.PERFORMANCE,
                    message="Pruning may increase inference latency",
                    parameter="enable_pruning",
                    current_value=True,
                    suggested_value=False,
                    explanation="Pruning can add overhead during inference",
                    impact="May increase latency by 10-20%",
                )
            )

    def _validate_device_specific_constraints(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate device-specific constraints and capabilities."""

        device = config.get("target_device", "cpu")
        if device not in self.device_capabilities:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.COMPATIBILITY,
                    message=f"Unsupported device: {device}",
                    parameter="target_device",
                    current_value=device,
                    suggested_value="cpu",
                    explanation="Supported devices: "
                    + ", ".join(self.device_capabilities.keys()),
                )
            )
            return

        capabilities = self.device_capabilities[device]

        # Check quantization support
        quantize = config.get("quantize", "none")
        if quantize == "float16" and not capabilities["supports_fp16"]:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.COMPATIBILITY,
                    message=f"Device {device} doesn't support FP16 quantization",
                    parameter="quantize",
                    current_value="float16",
                    suggested_value="int8",
                    explanation=f"{device} only supports INT8 quantization",
                )
            )

        # Check fusion support
        if config.get("enable_fusion", False) and not capabilities["supports_fusion"]:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.COMPATIBILITY,
                    message=f"Device {device} doesn't support operation fusion",
                    parameter="enable_fusion",
                    current_value=True,
                    suggested_value=False,
                    explanation=f"{device} doesn't have hardware support for operation fusion",
                )
            )

        # Check memory constraints
        memory_limit = config.get("memory_limit")
        if memory_limit:
            try:
                memory_mb = float(memory_limit)
                max_memory = capabilities["max_memory_mb"]
                if memory_mb > max_memory:
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category=ValidationCategory.COMPATIBILITY,
                            message=f"Memory limit {memory_mb}MB exceeds device capacity {max_memory}MB",
                            parameter="memory_limit",
                            current_value=memory_mb,
                            suggested_value=max_memory,
                            explanation=f"{device} has limited memory capacity",
                        )
                    )
            except (ValueError, TypeError):
                pass  # Already handled in basic validation

        # Check buffer size recommendations
        buffer_size = config.get("buffer_size")
        if buffer_size:
            try:
                buffer_num = int(buffer_size)
                recommended = capabilities["recommended_buffer_size"]
                if buffer_num > recommended * 2:
                    result.warnings.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category=ValidationCategory.PERFORMANCE,
                            message=f"Buffer size {buffer_num} may be too large for {device}",
                            parameter="buffer_size",
                            current_value=buffer_num,
                            suggested_value=recommended,
                            explanation=f"Recommended buffer size for {device} is {recommended}",
                            impact="May cause memory pressure and slower performance",
                        )
                    )
            except (ValueError, TypeError):
                pass  # Already handled in basic validation

    def _validate_model_format_compatibility(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate model format compatibility with other parameters."""

        model_path = config.get("model")
        if not model_path:
            return

        # Extract model format
        model_ext = os.path.splitext(model_path)[1].lower()
        if model_ext not in self.model_format_compatibility:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.COMPATIBILITY,
                    message=f"Unsupported model format: {model_ext}",
                    parameter="model",
                    current_value=model_path,
                    explanation="Supported formats: "
                    + ", ".join(self.model_format_compatibility.keys()),
                )
            )
            return

        compatibility = self.model_format_compatibility[model_ext]

        # Check quantization compatibility
        quantize = config.get("quantize", "none")
        if quantize == "int8" and not compatibility["quantize_int8"]:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.COMPATIBILITY,
                    message=f"Model format {model_ext} doesn't support INT8 quantization",
                    parameter="quantize",
                    current_value="int8",
                    suggested_value="none",
                    explanation=f"{model_ext} models cannot be quantized to INT8",
                )
            )

        if quantize == "float16" and not compatibility["quantize_float16"]:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.COMPATIBILITY,
                    message=f"Model format {model_ext} doesn't support FP16 quantization",
                    parameter="quantize",
                    current_value="float16",
                    suggested_value="none",
                    explanation=f"{model_ext} models cannot be quantized to FP16",
                )
            )

        # Check fusion compatibility
        if config.get("enable_fusion", False) and not compatibility["fusion"]:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.COMPATIBILITY,
                    message=f"Model format {model_ext} doesn't support operation fusion",
                    parameter="enable_fusion",
                    current_value=True,
                    suggested_value=False,
                    explanation=f"{model_ext} models cannot use operation fusion",
                )
            )

        # Check device recommendations
        device = config.get("target_device", "cpu")
        recommended_devices = compatibility["recommended_for"]
        if device not in recommended_devices:
            result.warnings.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.PERFORMANCE,
                    message=f"Model format {model_ext} is not optimal for {device}",
                    parameter="target_device",
                    current_value=device,
                    suggested_value=recommended_devices[0],
                    explanation=f"{model_ext} models work best on: {', '.join(recommended_devices)}",
                    impact="May result in suboptimal performance",
                )
            )

    def _estimate_performance_impact(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Estimate performance impact of configuration choices."""

        impact = {
            "size_reduction_percent": 0.0,
            "speed_improvement_factor": 1.0,
            "memory_reduction_percent": 0.0,
            "accuracy_impact_percent": 0.0,
            "confidence": 0.8,
        }

        # Quantization impact
        quantize = config.get("quantize", "none")
        if quantize == "int8":
            impact["size_reduction_percent"] += 75.0
            impact["speed_improvement_factor"] *= 2.0
            impact["memory_reduction_percent"] += 75.0
            impact["accuracy_impact_percent"] -= 3.0
        elif quantize == "float16":
            impact["size_reduction_percent"] += 50.0
            impact["speed_improvement_factor"] *= 1.5
            impact["memory_reduction_percent"] += 50.0
            impact["accuracy_impact_percent"] -= 1.0

        # Fusion impact
        if config.get("enable_fusion", False):
            impact["speed_improvement_factor"] *= 1.2
            impact["memory_reduction_percent"] += 10.0

        # Pruning impact
        if config.get("enable_pruning", False):
            sparsity = config.get("pruning_sparsity", 0.5)
            impact["size_reduction_percent"] += sparsity * 50.0
            impact["memory_reduction_percent"] += sparsity * 50.0
            impact["speed_improvement_factor"] *= 1.0 + sparsity * 0.3
            impact["accuracy_impact_percent"] -= sparsity * 2.0

        # Device-specific adjustments
        device = config.get("target_device", "cpu")
        if device in self.device_capabilities:
            tier = self.device_capabilities[device]["performance_tier"]
            if tier == "ultra_low":
                impact["speed_improvement_factor"] *= 0.8
                impact["confidence"] *= 0.9
            elif tier == "ultra_high":
                impact["speed_improvement_factor"] *= 1.3
                impact["confidence"] *= 1.1

        result.estimated_performance_impact = impact

    def _calculate_compatibility_score(self, result: ValidationResult) -> None:
        """Calculate overall compatibility score."""

        total_issues = len(result.issues) + len(result.warnings)
        if total_issues == 0:
            result.compatibility_score = 1.0
            return

        # Weight errors more heavily than warnings
        error_weight = 0.8
        warning_weight = 0.2

        error_score = len(result.issues) * error_weight
        warning_score = len(result.warnings) * warning_weight

        total_score = error_score + warning_score
        result.compatibility_score = max(0.0, 1.0 - (total_score / 10.0))

    def suggest_corrections(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest corrected configuration based on validation results.

        Args:
            config: Original configuration

        Returns:
            Dictionary with suggested corrections
        """
        result = self.validate_config(config)
        corrected_config = config.copy()

        # Apply error corrections
        for issue in result.issues:
            if issue.suggested_value is not None and issue.parameter:
                corrected_config[issue.parameter] = issue.suggested_value

        # Apply warning corrections (optional)
        for issue in result.warnings:
            if issue.suggested_value is not None and issue.parameter:
                # Only apply if it's a clear improvement
                if issue.category == ValidationCategory.PERFORMANCE:
                    corrected_config[issue.parameter] = issue.suggested_value

        return corrected_config

    def get_default_config(self, device: str = "cpu") -> Dict[str, Any]:
        """Get a default configuration optimized for a specific device.

        Args:
            device: Target device

        Returns:
            Default configuration dictionary
        """
        if device not in self.device_capabilities:
            device = "cpu"

        capabilities = self.device_capabilities[device]

        return {
            "model": "model.tflite",
            "quantize": "int8" if capabilities["supports_int8"] else "none",
            "target_device": device,
            "enable_fusion": capabilities["supports_fusion"],
            "enable_pruning": capabilities["supports_pruning"],
            "pruning_sparsity": 0.3,
            "buffer_size": capabilities["recommended_buffer_size"],
            "memory_limit": capabilities["max_memory_mb"] // 4,  # Conservative default
            "optimize_for": "balanced",
            "input_stream": "file",
            "deploy_path": "/models/",
        }


def validate_edgeflow_config_static(config: Dict[str, Any]) -> ValidationResult:
    """Main function for static validation of EdgeFlow configurations.

    Args:
        config: Parsed EdgeFlow configuration dictionary

    Returns:
        ValidationResult with detailed validation information
    """
    validator = EdgeFlowStaticValidator()
    return validator.validate_config(config)


def suggest_config_corrections(config: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest corrections for a configuration.

    Args:
        config: Original configuration

    Returns:
        Corrected configuration
    """
    validator = EdgeFlowStaticValidator()
    return validator.suggest_corrections(config)


def get_default_config(device: str = "cpu") -> Dict[str, Any]:
    """Get a default configuration for a device.

    Args:
        device: Target device

    Returns:
        Default configuration
    """
    validator = EdgeFlowStaticValidator()
    return validator.get_default_config(device)


if __name__ == "__main__":
    # Test the validator
    test_configs = [
        {
            "model": "mobilenet_v2.tflite",
            "quantize": "int8",
            "target_device": "raspberry_pi",
            "enable_fusion": True,
            "buffer_size": 32,
            "memory_limit": 64,
            "optimize_for": "latency",
        },
        {
            "model": "model.pth",
            "quantize": "int8",
            "target_device": "cortex_m4",
            "enable_fusion": True,
        },
        {"model": "model.h5", "quantize": "float16", "target_device": "cortex_m4"},
    ]

    validator = EdgeFlowStaticValidator()

    for i, config in enumerate(test_configs):
        print(f"\n=== Test Configuration {i+1} ===")
        print(f"Config: {config}")

        result = validator.validate_config(config)
        print(f"Valid: {result.is_valid}")
        print(f"Compatibility Score: {result.compatibility_score:.2f}")

        if result.issues:
            print("Errors:")
            for issue in result.issues:
                print(f"  - {issue.message}")
                if issue.suggested_value:
                    print(f"    Suggested: {issue.suggested_value}")

        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  - {warning.message}")

        print(f"Performance Impact: {result.estimated_performance_impact}")

        # Show corrections
        corrected = validator.suggest_corrections(config)
        if corrected != config:
            print(f"Corrected Config: {corrected}")
