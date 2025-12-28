"""Semantic validation passes for EdgeFlow IR/config.

This module provides semantic checks at multiple levels:
- Configuration-level validation for quick feedback
- Shape and type inference and validation
- Device compatibility checks
- Performance and resource constraint validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Type aliases
Shape = List[int]
TensorSpec = Dict[str, Any]


@dataclass
class Diagnostic:
    """A diagnostic message from the semantic validator."""

    code: str
    severity: str  # error | warning | info
    message: str
    hint: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class ShapeInferenceError(Exception):
    """Raised when shape inference fails."""

    pass


class SemanticValidator:
    """Validates model configurations against semantic rules and constraints."""

    # Supported operations and their parameter constraints
    SUPPORTED_OPS: Dict[str, Dict[str, Any]] = {
        "Conv2D": {
            "required_params": ["filters", "kernel_size"],
            "optional_params": {
                "strides": (1, 1),
                "padding": "valid",
                "activation": None,
            },
            "input_rank": 4,
            "output_rank": 4,
        },
        "Dense": {
            "required_params": ["units"],
            "optional_params": {"activation": None, "use_bias": True},
            "input_rank": 2,
            "output_rank": 2,
        },
        "ReLU": {
            "required_params": [],
            "optional_params": {},
            "input_rank": None,  # Any rank
            "output_rank": None,  # Same as input
        },
        "MaxPool2D": {
            "required_params": ["pool_size"],
            "optional_params": {"strides": None, "padding": "valid"},
            "input_rank": 4,
            "output_rank": 4,
        },
        "Flatten": {
            "required_params": [],
            "optional_params": {},
            "input_rank": None,  # Any rank
            "output_rank": 2,  # [batch_size, -1]
        },
    }

    # Supported quantization precisions
    SUPPORTED_QUANTIZATION = {"int8", "uint8", "float16", "float32"}

    def __init__(self, device_registry: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the semantic validator.

        Args:
            device_registry: Dictionary mapping device types to their capabilities and constraints
        """
        self.device_registry = device_registry or {}

    def validate_config(self, config: Dict[str, Any]) -> List[Diagnostic]:
        """Validate a model configuration.

        Args:
            config: The model configuration to validate

        Returns:
            List of diagnostic messages
        """
        # Initialize diagnostics list
        diagnostics: List[Diagnostic] = []

        # Basic structure validation
        if not isinstance(config, dict):
            return [
                self._create_diagnostic(
                    "invalid_config", "error", "Configuration must be a dictionary"
                )
            ]

        # Store the config for reference
        self.config = config

        # Validate model section
        if "model" not in config:
            return [
                self._create_diagnostic(
                    "missing_model_section",
                    "error",
                    "Missing 'model' section in configuration",
                )
            ]

        model_config = config["model"]

        # Store the model config for shape inference
        self.model_config = model_config

        # Store input shape if available
        if "input_shape" in model_config:
            self.input_shape = model_config["input_shape"]

        # Validate target device
        target_device = config.get("target_device", {})
        if not target_device:
            diagnostics.append(
                self._create_diagnostic(
                    "no_target_device", "warning", "No target device specified"
                )
            )

        # Validate model operations
        if "operations" in model_config:
            operations = model_config["operations"]
            if not isinstance(operations, list):
                diagnostics.append(
                    self._create_diagnostic(
                        "invalid_operations_type",
                        "error",
                        "Model operations must be a list",
                    )
                )
            else:
                # Add input shape to the first operation if it exists
                if operations and "input_shape" in model_config:
                    operations[0]["input_shape"] = model_config["input_shape"]

                # Validate operations and perform shape inference
                diagnostics.extend(self._validate_operations(operations, target_device))

                # Check if output shape matches the expected output shape
                if (
                    operations
                    and "output_shape" in model_config
                    and operations[-1].get("_output_shape")
                ):
                    last_op = operations[-1]
                    expected_output = model_config["output_shape"]
                    actual_output = last_op["_output_shape"]

                    if actual_output is None:
                        diagnostics.append(
                            self._create_diagnostic(
                                "output_shape_inference_failed",
                                "error",
                                "Could not infer output shape for the model",
                                hint="Check that all operations have valid parameters and input shapes",
                            )
                        )
                    # Compare shapes ignoring batch size
                    elif expected_output[1:] != actual_output[1:]:
                        diagnostics.append(
                            self._create_diagnostic(
                                "output_shape_mismatch",
                                "error",
                                f"Model output shape {actual_output} does not match expected output shape {expected_output}",
                                hint="Check that the model architecture produces the expected output shape",
                            )
                        )

        # Validate optimization settings
        if "optimization" in config:
            optimization = config["optimization"]
            if not isinstance(optimization, dict):
                diagnostics.append(
                    self._create_diagnostic(
                        "invalid_optimization_type",
                        "error",
                        "Optimization config must be a dictionary",
                    )
                )
            else:
                diagnostics.extend(self._validate_optimization(optimization))

        # Validate device compatibility if target device is specified
        if target_device and "type" in target_device:
            device_type = target_device["type"]
            if device_type in self.device_registry:
                device_spec = self.device_registry[device_type]
                diagnostics.extend(
                    self._validate_device_compatibility(
                        model_config, target_device, self.device_registry[device_type]
                    )
                )
            else:
                diagnostics.append(
                    self._create_diagnostic(
                        "unknown_device",
                        "warning",
                        f"Device type '{device_type}' not found in registry",
                    )
                )

        return diagnostics

    def _validate_operations(
        self, operations: List[Dict[str, Any]], target_device: Dict[str, Any]
    ) -> List[Diagnostic]:
        """Validate the operations in the model."""
        diagnostics: List[Diagnostic] = []

        if not operations:
            return [
                self._create_diagnostic(
                    "empty_operations",
                    "error",
                    "Model must have at least one operation",
                )
            ]

        # Track input shapes if available
        input_shape = None
        if hasattr(self, "model_config") and "input_shape" in self.model_config:
            input_shape = self.model_config["input_shape"]
            self.input_shape = input_shape

            # Add input shape to the first operation
            if operations:
                operations[0]["input_shape"] = input_shape

        # Check for unsupported operations and validate parameters
        for i, op in enumerate(operations):
            op_type = op.get("type")
            if not op_type:
                diagnostics.append(
                    self._create_diagnostic(
                        "missing_op_type",
                        "error",
                        f"Operation at index {i} is missing a type",
                        location={"operation_index": i},
                    )
                )
                continue

            if op_type not in self.SUPPORTED_OPS:
                diagnostics.append(
                    self._create_diagnostic(
                        "unsupported_operation",
                        "error",
                        f"Unsupported operation type: {op_type}",
                        hint=f"Supported operations are: {', '.join(self.SUPPORTED_OPS.keys())}",
                        location={"operation_index": i, "operation_type": op_type},
                    )
                )
                continue

            # Check required parameters
            op_spec = self.SUPPORTED_OPS[op_type]
            for param in op_spec["required_params"]:
                if param not in op:
                    diagnostics.append(
                        self._create_diagnostic(
                            "missing_required_param",
                            "error",
                            f"Operation '{op_type}' is missing required parameter: {param}",
                            location={"operation_index": i, "parameter": param},
                        )
                    )

            # Validate parameter types and values
            for param, value in op.items():
                if param == "type":
                    continue

                # Validate kernel_size
                if param == "kernel_size":
                    if isinstance(value, int):
                        if value <= 0:
                            diagnostics.append(
                                self._create_diagnostic(
                                    "invalid_kernel_size",
                                    "error",
                                    f"Kernel size must be positive, got {value}",
                                    location={"operation_index": i, "parameter": param},
                                )
                            )
                    elif isinstance(value, (list, tuple)):
                        if any(not isinstance(x, int) or x <= 0 for x in value):
                            diagnostics.append(
                                self._create_diagnostic(
                                    "invalid_kernel_size",
                                    "error",
                                    f"Kernel size dimensions must be positive integers, got {value}",
                                    location={"operation_index": i, "parameter": param},
                                )
                            )

                # Validate filters/units
                elif param in ["filters", "units"] and (
                    not isinstance(value, int) or value <= 0
                ):
                    diagnostics.append(
                        self._create_diagnostic(
                            f"invalid_{param}",
                            "error",
                            f"{param} must be a positive integer, got {value}",
                            location={"operation_index": i, "parameter": param},
                        )
                    )

        # Perform shape inference
        try:
            self._infer_shapes(operations, target_device)

            # Check for shape mismatches after inference
            for i in range(len(operations)):
                op = operations[i]

                # Check for invalid input shapes in Conv2D operations
                if op.get("type") == "Conv2D" and "input_shape" in op:
                    input_shape = op["input_shape"]
                    if input_shape is not None and len(input_shape) != 4:
                        diagnostics.append(
                            self._create_diagnostic(
                                "invalid_input_shape",
                                "error",
                                f"Conv2D operation at index {i} expects 4D input (batch, height, width, channels), "
                                f"got shape {input_shape}",
                                location={
                                    "operation_index": i,
                                    "operation_type": "Conv2D",
                                },
                            )
                        )

            # Check for shape mismatches between operations
            for i in range(1, len(operations)):
                prev_op = operations[i - 1]
                curr_op = operations[i]

                # Skip if we don't have shape information for either operation
                if "_output_shape" not in prev_op or "input_shape" not in curr_op:
                    continue

                prev_output = prev_op["_output_shape"]
                curr_input = curr_op["input_shape"]

                # Skip if we don't have valid shape information
                if prev_output is None or curr_input is None:
                    continue

                # Handle Dense layer shape validation
                if curr_op.get("type") == "Dense":
                    # Calculate the expected input size for the Dense layer
                    expected_units = curr_op.get("units")
                    if expected_units is None:
                        continue

                    # Check if the input needs to be flattened
                    if len(prev_output) > 2:
                        # Calculate the total number of elements (excluding batch dimension)
                        import numpy as np

                        total_elements = int(np.prod(prev_output[1:]))

                        # Check if we need a Flatten layer
                        if (
                            total_elements != expected_units
                            and curr_op.get("input_shape", [None])[1] != total_elements
                        ):
                            # Check if there's already a shape mismatch error for this operation
                            if not any(
                                d.code == "shape_mismatch"
                                and d.location is not None
                                and d.location.get("next_operation_index") == i
                                for d in diagnostics
                            ):
                                diagnostics.append(
                                    self._create_diagnostic(
                                        "shape_mismatch",
                                        "error",
                                        f"Dense layer at index {i} expects input shape [batch_size, {expected_units}], "
                                        f"but previous layer output shape is {prev_output}",
                                        hint=f"Add a Flatten layer before the Dense layer or adjust the layer configuration",
                                        location={
                                            "operation_index": i - 1,
                                            "operation_type": prev_op.get("type"),
                                            "next_operation_index": i,
                                            "next_operation_type": "Dense",
                                            "output_shape": prev_output,
                                            "expected_input_shape": [
                                                None,
                                                expected_units,
                                            ],
                                        },
                                    )
                                )
                            continue

                # For non-Dense layers or when shapes are already 2D
                if prev_output != curr_input:
                    # Check if there's already a shape mismatch error for this operation
                    if not any(
                        d.code == "shape_mismatch"
                        and d.location is not None
                        and d.location.get("next_operation_index") == i
                        for d in diagnostics
                    ):
                        diagnostics.append(
                            self._create_diagnostic(
                                "shape_mismatch",
                                "error",
                                f"Shape mismatch between operations {i-1} ({prev_op.get('type')}) and {i} ({curr_op.get('type')}): "
                                f"{prev_output} vs {curr_input}",
                                hint="Check that the output shape of one operation matches the input shape of the next",
                                location={
                                    "operation_index": i - 1,
                                    "operation_type": prev_op.get("type"),
                                    "next_operation_index": i,
                                    "next_operation_type": curr_op.get("type"),
                                    "output_shape": prev_output,
                                    "expected_input_shape": curr_input,
                                },
                            )
                        )

        except ShapeInferenceError as e:
            diagnostics.append(
                self._create_diagnostic(
                    "shape_inference_error",
                    "error",
                    f"Shape inference failed: {str(e)}",
                    hint="Check that the model architecture is valid and all shapes are compatible",
                )
            )

        return diagnostics

    def _validate_optimization(self, optimization: Dict[str, Any]) -> List[Diagnostic]:
        """Validate optimization settings with enhanced validation rules.

        Args:
            optimization: Dictionary containing optimization settings

        Returns:
            List of Diagnostic objects with validation results
        """
        diagnostics: List[Diagnostic] = []

        if not isinstance(optimization, dict):
            return [
                self._create_diagnostic(
                    "invalid_optimization_type",
                    "error",
                    "Optimization configuration must be a dictionary",
                )
            ]

        # Define supported optimization techniques and their requirements
        SUPPORTED_OPTIMIZATIONS: Dict[str, Dict[str, Any]] = {
            "quantization": {
                "type": dict,
                "required_fields": ["enabled"],
                "params": {
                    "int8": {"calibration_required": True, "hybrid_supported": True},
                    "int4": {"calibration_required": True, "hybrid_required": True},
                    "float16": {"calibration_required": False},
                    "bfloat16": {
                        "calibration_required": False,
                        "device_support": ["gpu", "tpu"],
                    },
                },
            },
            "pruning": {
                "type": dict,
                "params": {
                    "sparsity": (float, 0.0, 0.95),
                    "schedule": ["constant", "polynomial", "exponential"],
                    "frequency": (int, 1, 100),
                },
            },
            "optimization_level": {
                "type": (int, str),
                "values": ["0", "1", "2", "3", 0, 1, 2, 3],
            },
            "fusion": {"type": (bool, dict), "params": ["enabled", "patterns"]},
            "memory_optimization": {
                "type": (bool, dict),
                "params": ["enabled", "offload_activations"],
            },
            "mixed_precision": {
                "type": (bool, dict),
                "params": ["enabled", "loss_scale"],
            },
            "distillation": {
                "type": dict,
                "required_fields": ["teacher_model"],
                "params": {
                    "temperature": (float, 0.1, 10.0),
                    "alpha": (float, 0.0, 1.0),
                },
            },
        }

        # Check for unsupported optimization techniques
        for key in optimization:
            if key not in SUPPORTED_OPTIMIZATIONS:
                diagnostics.append(
                    self._create_diagnostic(
                        "unsupported_optimization",
                        "warning",
                        f"Unsupported optimization technique: {key}",
                        hint=f"Supported techniques: {', '.join(SUPPORTED_OPTIMIZATIONS.keys())}",
                    )
                )

        # Validate each optimization technique
        for opt_name, opt_spec in SUPPORTED_OPTIMIZATIONS.items():
            if opt_name not in optimization:
                continue

            opt_value = optimization[opt_name]

            # Check type
            expected_type = opt_spec["type"]
            if not isinstance(opt_value, expected_type):
                diagnostics.append(
                    self._create_diagnostic(
                        f"invalid_{opt_name}_type",
                        "error",
                        f"{opt_name} must be of type {expected_type}",
                        context={"actual_type": type(opt_value).__name__},
                    )
                )
                continue

            # Handle specific optimization validations
            if opt_name == "quantization":
                self._validate_quantization(opt_value, diagnostics)

            elif opt_name == "pruning" and isinstance(opt_value, dict):
                self._validate_pruning(opt_value, diagnostics)

            elif opt_name == "optimization_level":
                if str(opt_value) not in [str(x) for x in opt_spec["values"]]:
                    diagnostics.append(
                        self._create_diagnostic(
                            "invalid_optimization_level",
                            "error",
                            f"Invalid optimization level: {opt_value}",
                            hint=f"Must be one of: {', '.join(map(str, opt_spec['values']))}",
                        )
                    )

            # Check required fields for dictionary-based optimizations
            if isinstance(opt_value, dict) and "required_fields" in opt_spec:
                for field in opt_spec["required_fields"]:
                    if field not in opt_value:
                        diagnostics.append(
                            self._create_diagnostic(
                                f"missing_required_field_{opt_name}",
                                "error",
                                f"Missing required field '{field}' for {opt_name}",
                            )
                        )

        return diagnostics

    def _validate_quantization(
        self, quant_config: Dict[str, Any], diagnostics: List[Diagnostic]
    ) -> None:
        """Validate quantization configuration."""
        if not quant_config.get("enabled", False):
            return

        # Check precision
        precision = quant_config.get("precision", "int8")
        if precision not in self.SUPPORTED_QUANTIZATION:
            diagnostics.append(
                self._create_diagnostic(
                    "unsupported_quantization_precision",
                    "error",
                    f"Unsupported quantization precision: {precision}",
                    hint=f"Supported precisions: {', '.join(sorted(self.SUPPORTED_QUANTIZATION))}",
                    context={"provided_precision": precision},
                )
            )

        # Check calibration data requirements
        if precision in ["int8", "int4"] and not quant_config.get("calibration_data"):
            diagnostics.append(
                self._create_diagnostic(
                    "missing_calibration_data",
                    "warning" if precision == "int8" else "error",
                    f"Calibration data is required for {precision} quantization",
                    hint="Provide representative calibration data for better accuracy",
                )
            )

        # Check hybrid quantization requirements for int4
        if precision == "int4" and not quant_config.get("use_hybrid_quantization"):
            diagnostics.append(
                self._create_diagnostic(
                    "int4_hybrid_required",
                    "error",
                    "int4 quantization requires hybrid quantization",
                    hint="Set 'use_hybrid_quantization': True in quantization config",
                )
            )

    def _validate_pruning(
        self, prune_config: Dict[str, Any], diagnostics: List[Diagnostic]
    ) -> None:
        """Validate pruning configuration."""
        if not prune_config.get("enabled", False):
            return

        # Check sparsity
        sparsity = prune_config.get("sparsity", 0.5)
        if not isinstance(sparsity, (int, float)) or not (0 <= sparsity <= 0.95):
            diagnostics.append(
                self._create_diagnostic(
                    "invalid_pruning_sparsity",
                    "error",
                    f"Invalid sparsity value: {sparsity}",
                    hint="Sparsity must be between 0.0 and 0.95",
                )
            )

        # Check schedule
        schedule = prune_config.get("schedule", "constant")
        if schedule not in ["constant", "polynomial", "exponential"]:
            diagnostics.append(
                self._create_diagnostic(
                    "invalid_pruning_schedule",
                    "warning",
                    f"Unsupported pruning schedule: {schedule}",
                    hint="Supported schedules: constant, polynomial, exponential",
                )
            )

        # Check frequency
        frequency = prune_config.get("frequency", 100)
        if not isinstance(frequency, int) or frequency < 1:
            diagnostics.append(
                self._create_diagnostic(
                    "invalid_pruning_frequency",
                    "error",
                    f"Invalid pruning frequency: {frequency}",
                    hint="Frequency must be a positive integer",
                )
            )

    def _estimate_optimization_impact(
        self, optimization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate the impact of optimizations on model performance.

        Returns:
            Dictionary with estimated metrics (compression_ratio, speedup, accuracy_drop)
        """
        impact = {
            "compression_ratio": 1.0,
            "speedup": 1.0,
            "accuracy_drop": 0.0,
            "memory_reduction": 1.0,
        }

        if not optimization:
            return impact

        # Estimate impact of each optimization
        if "quantization" in optimization and optimization["quantization"].get(
            "enabled"
        ):
            precision = optimization["quantization"].get("precision", "int8")
            if precision == "int8":
                impact["compression_ratio"] *= 0.25  # 4x smaller
                impact["speedup"] *= 2.0  # 2x faster
                impact["accuracy_drop"] += 0.01  # 1% accuracy drop
            elif precision == "int4":
                impact["compression_ratio"] *= 0.125  # 8x smaller
                impact["speedup"] *= 4.0  # 4x faster
                impact["accuracy_drop"] += 0.05  # 5% accuracy drop
            elif precision == "float16":
                impact["compression_ratio"] *= 0.5  # 2x smaller
                impact["speedup"] *= 1.5  # 1.5x faster
                impact["accuracy_drop"] += 0.001  # 0.1% accuracy drop

        if "pruning" in optimization and optimization["pruning"].get("enabled"):
            sparsity = optimization["pruning"].get("sparsity", 0.5)
            impact["compression_ratio"] *= 1 - sparsity
            impact["speedup"] *= 1 + sparsity * 0.5  # Up to 1.5x faster
            impact["accuracy_drop"] += sparsity * 0.1  # Up to 10% accuracy drop

        if "fusion" in optimization and optimization["fusion"].get("enabled", True):
            impact["speedup"] *= 1.2  # 20% faster

        if "memory_optimization" in optimization and optimization[
            "memory_optimization"
        ].get("enabled", False):
            impact["memory_reduction"] *= 0.5  # 50% memory reduction

        return impact

    def _validate_device_compatibility(
        self,
        model_config: Dict[str, Any],
        target_device: Dict[str, Any],
        device_spec: Dict[str, Any],
    ) -> List[Diagnostic]:
        """Validate model compatibility with target device."""
        diagnostics: List[Diagnostic] = []

        # Check if all operations are supported by the device
        if "operations" in model_config and "supported_ops" in device_spec.get(
            "constraints", {}
        ):
            supported_ops = set(device_spec["constraints"]["supported_ops"])
            for i, op in enumerate(model_config["operations"]):
                op_type = op.get("type")
                if op_type and op_type not in supported_ops:
                    diagnostics.append(
                        self._create_diagnostic(
                            "unsupported_operation_for_device",
                            "error",
                            f"Operation '{op_type}' is not supported on the target device",
                            hint=f"Supported operations are: {', '.join(supported_ops)}",
                            location={"operation_index": i, "operation_type": op_type},
                        )
                    )

        # Check memory constraints
        if "constraints" in device_spec:
            constraints = device_spec["constraints"]
            if "max_memory_mb" in constraints and "operations" in model_config:
                # Simple memory estimation (very basic, should be replaced with actual model analysis)
                estimated_memory = self._estimate_memory_usage(model_config)
                max_memory = constraints["max_memory_mb"]

                if estimated_memory > max_memory * 0.9:  # Leave 10% margin
                    diagnostics.append(
                        self._create_diagnostic(
                            "memory_constraint_violation",
                            "warning" if estimated_memory <= max_memory else "error",
                            f"Estimated memory usage ({estimated_memory}MB) is close to device limit ({max_memory}MB)",
                            hint="Consider reducing model size or using model optimization techniques",
                            context={
                                "estimated_memory_mb": estimated_memory,
                                "max_memory_mb": max_memory,
                            },
                        )
                    )

        return diagnostics

    def _infer_shapes(
        self, operations: List[Dict[str, Any]], target_device: Dict[str, Any]
    ) -> None:
        """Perform shape inference on the model.

        Args:
            operations: List of operations in the model
            target_device: Target device configuration

        Raises:
            ShapeInferenceError: If shape inference fails
        """
        if not operations:
            return

        # Initialize with input shape if available
        input_shape = None
        if hasattr(self, "input_shape"):
            input_shape = self.input_shape

        current_shape = input_shape

        for i, op in enumerate(operations):
            op_type = op.get("type")
            if not op_type or op_type not in self.SUPPORTED_OPS:
                continue

            op_spec = self.SUPPORTED_OPS[op_type]

            # Store the input shape for this operation
            op["input_shape"] = current_shape

            # Skip if we don't have a shape to work with
            if current_shape is None:
                op["_output_shape"] = None
                continue

            # For Conv2D, check input shape and generate appropriate error
            if op_type == "Conv2D":
                if len(current_shape) != 4:
                    op["_output_shape"] = None
                    # Don't raise an error, just set the output shape to None
                    # The validation will be handled in _validate_operations
                    continue

            # For Dense layers with >2D input, we'll handle this in _validate_operations
            elif op_type == "Dense" and len(current_shape) > 2:
                op["_output_shape"] = None
                continue

            # Shape inference for each operation type
            try:
                if op_type == "Conv2D":
                    filters = op.get("filters")
                    kernel_size = op.get("kernel_size")
                    strides = op.get("strides", (1, 1))
                    padding = op.get("padding", "valid")

                    if (
                        not all(isinstance(x, int) for x in current_shape[1:])
                        or not filters
                        or not kernel_size
                    ):
                        raise ShapeInferenceError(
                            f"Invalid parameters for Conv2D: filters={filters}, kernel_size={kernel_size}"
                        )

                    if isinstance(kernel_size, int):
                        kernel_size = (kernel_size, kernel_size)
                    if isinstance(strides, int):
                        strides = (strides, strides)

                    batch, h_in, w_in, channels_in = current_shape

                    if padding.lower() == "same":
                        h_out = (h_in + strides[0] - 1) // strides[0]
                        w_out = (w_in + strides[1] - 1) // strides[1]
                    else:  # 'valid' padding
                        h_out = (h_in - kernel_size[0]) // strides[0] + 1
                        w_out = (w_in - kernel_size[1]) // strides[1] + 1

                        if h_out <= 0 or w_out <= 0:
                            raise ShapeInferenceError(
                                f"Output shape would be non-positive: ({h_out}, {w_out}). "
                                f"Input shape: {current_shape}, kernel_size: {kernel_size}, strides: {strides}"
                            )

                    current_shape = [batch, h_out, w_out, filters]

                elif op_type == "Dense":
                    units = op.get("units")
                    if not isinstance(units, int) or units <= 0:
                        raise ShapeInferenceError(
                            f"Invalid units value for Dense layer: {units}"
                        )

                    if len(current_shape) != 2:
                        # If input is not 2D, we need to check if it can be flattened
                        total_elements = 1
                        for dim in current_shape[1:]:  # Skip batch dimension
                            if not isinstance(dim, int) or dim <= 0:
                                raise ShapeInferenceError(
                                    f"Cannot infer shape for Dense layer with input shape {current_shape}"
                                )
                            total_elements *= dim

                        current_shape = [current_shape[0], total_elements]  # Flatten

                    current_shape = [
                        current_shape[0],
                        units,
                    ]  # Apply Dense transformation

                elif op_type == "MaxPool2D":
                    pool_size = op.get("pool_size", 2)
                    strides = op.get("strides", pool_size)
                    padding = op.get("padding", "valid")

                    if len(current_shape) != 4:
                        raise ShapeInferenceError(
                            f"MaxPool2D requires 4D input, got shape {current_shape}"
                        )

                    if isinstance(pool_size, int):
                        pool_size = (pool_size, pool_size)
                    if isinstance(strides, int):
                        strides = (strides, strides)

                    batch, h_in, w_in, channels = current_shape

                    if padding.lower() == "same":
                        h_out = (h_in + strides[0] - 1) // strides[0]
                        w_out = (w_in + strides[1] - 1) // strides[1]
                    else:  # 'valid' padding
                        h_out = (h_in - pool_size[0]) // strides[0] + 1
                        w_out = (w_in - pool_size[1]) // strides[1] + 1

                        if h_out <= 0 or w_out <= 0:
                            raise ShapeInferenceError(
                                f"Output shape would be non-positive: ({h_out}, {w_out}). "
                                f"Input shape: {current_shape}, pool_size: {pool_size}, strides: {strides}"
                            )

                    current_shape = [batch, h_out, w_out, channels]

                elif op_type == "Flatten":
                    if len(current_shape) < 2:
                        raise ShapeInferenceError(
                            f"Cannot flatten shape {current_shape} with less than 2 dimensions"
                        )

                    # Calculate total elements in all dimensions except batch
                    total_elements = 1
                    for dim in current_shape[1:]:  # Skip batch dimension
                        if not isinstance(dim, int) or dim <= 0:
                            raise ShapeInferenceError(
                                f"Cannot flatten shape {current_shape} with non-positive or unknown dimensions"
                            )
                        total_elements *= dim

                    current_shape = [current_shape[0], total_elements]

            except Exception as e:
                raise ShapeInferenceError(
                    f"Error in shape inference for {op_type} operation at index {i}: {str(e)}"
                ) from e

            # Store the output shape in the operation
            op["_output_shape"] = current_shape

    def _estimate_memory_usage(self, model_config: Dict[str, Any]) -> float:
        """Estimate the memory usage of the model.

        Args:
            model_config: The model configuration

        Returns:
            Estimated memory usage in MB
        """
        # This is a placeholder implementation
        # A real implementation would analyze the model architecture, weights, and activations
        return 100.0  # Dummy value

    @staticmethod
    def _create_diagnostic(
        code: str,
        severity: str,
        message: str,
        hint: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Diagnostic:
        """Helper to create a diagnostic message."""
        return Diagnostic(
            code=code,
            severity=severity,
            message=message,
            hint=hint,
            location=location,
            context=context,
        )

    # ===== IR-LEVEL VALIDATION =====
    def validate_ir_graph(
        self, graph: Any, target_device: Optional[str] = None
    ) -> List[Diagnostic]:
        """Validate an IR graph (edgeflow IR or unified IR).

        Args:
            graph: IR graph instance (edgeflow_ir.IRGraph or unified_ir.UIRGraph)
            target_device: Optional device key to validate against.

        Returns:
            List of diagnostics (errors/warnings/info).
        """
        diags: List[Diagnostic] = []

        # Device capabilities (minimal exemplar). In production, load from registry.
        device = (
            target_device
            or getattr(graph, "metadata", {}).get("target_device")
            or "cpu"
        ).lower()
        caps = self._get_device_capabilities(device)

        # Basic connectivity and cycle checks
        diags.extend(self._validate_connectivity(graph))

        # Shape and dtype consistency
        diags.extend(self._validate_shapes_and_dtypes(graph))

        # Operator legality and parameter constraints
        diags.extend(self._validate_ops_and_params(graph, caps))

        # Resource/memory constraints (heuristic)
        diags.extend(self._validate_resources(graph, caps))

        return diags

    # ---- Helpers ----
    def _validate_connectivity(self, graph: Any) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        try:
            if hasattr(graph, "validate_graph"):
                ok, errors = graph.validate_graph()
                if not ok:
                    for err in errors:
                        severity = "error" if "cycle" in err.lower() else "warning"
                        diags.append(
                            Diagnostic(
                                code="IR201",
                                severity=severity,
                                message=f"Graph connectivity issue: {err}",
                            )
                        )
        except Exception as exc:  # noqa: BLE001
            diags.append(
                Diagnostic(
                    code="IR299",
                    severity="warning",
                    message=f"Connectivity validation failed: {exc}",
                )
            )

        # Dangling nodes: no inputs (not graph input) or no outputs (not graph output)
        try:
            graph_inputs = set(getattr(graph, "graph_inputs", []) or [])
            graph_outputs = set(getattr(graph, "graph_outputs", []) or [])

            if hasattr(graph, "nodes"):
                for node_id, node in getattr(graph, "nodes").items():
                    deps = list(getattr(node, "dependencies", []) or [])
                    outs = list(getattr(node, "dependents", []) or [])
                    # Fall back to canonical inputs/outputs if available
                    if hasattr(node, "inputs") and not deps:
                        deps = list(getattr(node, "inputs") or [])
                    if hasattr(node, "outputs") and not outs:
                        outs = list(getattr(node, "outputs") or [])

                    if not deps and node_id not in graph_inputs:
                        diags.append(
                            Diagnostic(
                                code="IR202",
                                severity="warning",
                                message="Dangling node has no inputs",
                                location={"node_id": node_id},
                            )
                        )
                    if not outs and node_id not in graph_outputs:
                        diags.append(
                            Diagnostic(
                                code="IR203",
                                severity="warning",
                                message="Dangling node has no outputs",
                                location={"node_id": node_id},
                            )
                        )
        except Exception:
            # Non-fatal; rely on validate_graph output
            pass

        return diags

    def _validate_shapes_and_dtypes(self, graph: Any) -> List[Diagnostic]:
        diags: List[Diagnostic] = []

        # Iterate edges depending on IR flavor
        edges: List[Tuple[str, str]] = []
        if hasattr(graph, "edges") and graph.edges:
            # edgeflow_ir stores List[Tuple[str, str]]; unified_ir stores triples
            first = graph.edges[0]
            if isinstance(first, tuple) and len(first) == 3:
                edges = [(e[0], e[1]) for e in graph.edges]
            else:
                edges = list(graph.edges)

        for src_id, dst_id in edges:
            try:
                src = graph.nodes[src_id]
                dst = graph.nodes[dst_id]
            except Exception:
                # Missing node references handled in connectivity checks
                continue

            # Shapes: prefer canonical fields; fall back to properties
            src_out_shapes = getattr(src, "output_shapes", None)
            dst_in_shapes = getattr(dst, "input_shapes", None)
            if not src_out_shapes:
                src_out_shapes = [
                    self._extract_shape_from_properties(src, "output_shape")
                ]
            if not dst_in_shapes:
                dst_in_shapes = [
                    self._extract_shape_from_properties(dst, "input_shape")
                ]

            if (
                src_out_shapes
                and dst_in_shapes
                and src_out_shapes[0]
                and dst_in_shapes[0]
            ):
                if not self._shapes_compatible(src_out_shapes[0], dst_in_shapes[0]):
                    diags.append(
                        Diagnostic(
                            code="IR301",
                            severity="error",
                            message=f"Shape mismatch {src_out_shapes[0]} -> {dst_in_shapes[0]}",
                            location={"from": src_id, "to": dst_id},
                            context={
                                "from_op": getattr(
                                    src, "op_type", getattr(src, "node_type", None)
                                ),
                                "to_op": getattr(
                                    dst, "op_type", getattr(dst, "node_type", None)
                                ),
                            },
                        )
                    )

            # Dtypes
            sdt = getattr(src, "dtype", None) or self._extract_dtype(src)
            ddt = getattr(dst, "dtype", None) or self._extract_dtype(dst)
            if sdt and ddt and sdt != ddt:
                diags.append(
                    Diagnostic(
                        code="IR302",
                        severity="warning",
                        message=f"Dtype mismatch {sdt} -> {ddt}",
                        location={"from": src_id, "to": dst_id},
                    )
                )

        return diags

    def _validate_ops_and_params(
        self, graph: Any, caps: Dict[str, Any]
    ) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        supported_ops = set(caps.get("supported_ops", []))
        limits = caps.get("limits", {})

        for node_id, node in getattr(graph, "nodes", {}).items():
            op = getattr(node, "op_type", None) or getattr(node, "node_type", None)
            op_name = getattr(op, "value", op)  # Enum or str

            # Check operator support if canonical string available
            if isinstance(op_name, str):
                if supported_ops and op_name.lower() not in {
                    o.lower() for o in supported_ops
                }:
                    diags.append(
                        Diagnostic(
                            code="IR401",
                            severity="error",
                            message=f"Unsupported operator for device: {op_name}",
                            location={"node_id": node_id},
                            hint="Replace op or target a different device",
                        )
                    )

            # Parameter constraints: example for Conv2D kernel size
            params = getattr(node, "params", None) or getattr(node, "properties", {})
            ks = params.get("kernel_size") if isinstance(params, dict) else None
            max_k = limits.get("max_kernel_size")
            if ks and max_k and isinstance(ks, (list, tuple)) and len(ks) == 2:
                if ks[0] > max_k or ks[1] > max_k:
                    diags.append(
                        Diagnostic(
                            code="IR402",
                            severity="error",
                            message=f"Kernel size {ks} exceeds device limit {max_k}",
                            location={"node_id": node_id},
                            hint="Reduce kernel_size or change target device",
                        )
                    )

        return diags

    def _validate_resources(self, graph: Any, caps: Dict[str, Any]) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        mem_limit = caps.get("limits", {}).get("memory_mb")
        if not mem_limit:
            return diags

        # Heuristic: estimate memory as small base + per-node cost
        node_count = len(getattr(graph, "nodes", {}))
        est_mb = 4.0 + 0.1 * node_count  # arbitrary simple heuristic
        if est_mb > mem_limit:
            diags.append(
                Diagnostic(
                    code="IR501",
                    severity="warning",
                    message=f"Estimated memory {est_mb:.1f}MB exceeds device limit {mem_limit}MB",
                    hint="Consider reducing model size or using a device with more memory",
                )
            )
        return diags

    # ---- Utility helpers ----
    def _extract_shape_from_properties(
        self, node: Any, key: str
    ) -> Optional[List[int]]:
        props = getattr(node, "properties", {}) or {}
        v = props.get(key)
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            try:
                return [int(x) for x in v]
            except Exception:
                return None
        if isinstance(v, str):
            try:
                return [
                    int(dim.strip()) if dim.strip().isdigit() else -1
                    for dim in v.split(",")
                ]
            except Exception:
                return None
        return None

    def _extract_dtype(self, node: Any) -> Optional[str]:
        # Try canonical dtype, fallback to properties["data_type"]
        dt = getattr(node, "dtype", None)
        if dt:
            return dt
        props = getattr(node, "properties", {}) or {}
        return props.get("data_type")

    def _shapes_compatible(self, out_shape: List[Any], in_shape: List[Any]) -> bool:
        # Simple compatibility check: exact or broadcast where 1 allowed
        if len(out_shape) != len(in_shape):
            return False
        for a, b in zip(out_shape, in_shape):
            if a == b:
                continue
            # Allow -1 as dynamic wildcard
            if a == -1 or b == -1:
                continue
            # Allow broadcasting from 1
            if a == 1 or b == 1:
                continue
            return False
        return True

    def _get_device_capabilities(self, device: str) -> Dict[str, Any]:
        # Merge registry-provided caps with defaults for multiple backends
        cpu_caps: Dict[str, Any] = {
            "supported_ops": [
                "input",
                "model",
                "output",
                "conv2d",
                "dense",
                "relu",
                "max_pool",
                "avg_pool",
                "quantize",
                "fusion",
                "schedule",
            ],
            "limits": {"max_kernel_size": 7, "memory_mb": 2048},
        }
        gpu_caps = {
            "supported_ops": cpu_caps["supported_ops"]
            + [
                "matmul",
                "batch_norm",
                "layer_norm",
            ],
            "limits": {"max_kernel_size": 11, "memory_mb": 8192},
        }
        tpu_caps = {
            "supported_ops": [
                "input",
                "output",
                "conv2d",
                "dense",
                "relu",
                "max_pool",
                "avg_pool",
                "quantize",
            ],
            "limits": {"max_kernel_size": 7, "memory_mb": 8192},
        }
        micro_caps = {
            "supported_ops": [
                "input",
                "output",
                "conv2d",
                "dense",
                "relu",
                "max_pool",
                "avg_pool",
            ],
            "limits": {"max_kernel_size": 5, "memory_mb": 64},
        }

        defaults_map = {
            "cpu": cpu_caps,
            "gpu": gpu_caps,
            "tpu": tpu_caps,
            "raspberry_pi": {
                **cpu_caps,
                "limits": {"max_kernel_size": 7, "memory_mb": 64},
            },
            "microcontroller": micro_caps,
        }
        defaults: Dict[str, Any] = defaults_map.get(device, cpu_caps)
        caps = dict(defaults)
        if self.device_registry and device in self.device_registry:
            # Shallow merge
            custom = self.device_registry[device]
            caps.update(custom)
            if "limits" in custom:
                caps["limits"].update(custom["limits"])  # type: ignore[index]
        return caps
