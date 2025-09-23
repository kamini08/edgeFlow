"""Framework-agnostic semantic validators for UIR graphs.

This module implements semantic validators that work on the unified intermediate
representation (UIR) to enforce domain-specific rules regardless of the source
framework. These validators check for shape compatibility, operator support,
and edge device constraints.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from unified_ir import (
    DataType,
    FrameworkType,
    OperationType,
    TensorInfo,
    TensorShape,
    UIRGraph,
    UIRNode,
    UIRValidator,
)

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationCategory(Enum):
    """Categories of validation issues."""

    SHAPE_COMPATIBILITY = "shape_compatibility"
    OPERATOR_SUPPORT = "operator_support"
    DEVICE_CONSTRAINTS = "device_constraints"
    DATA_TYPE_COMPATIBILITY = "data_type_compatibility"
    PERFORMANCE = "performance"
    SEMANTIC = "semantic"


@dataclass
class ValidationIssue:
    """Represents a validation issue with context and suggestions."""

    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    node_id: Optional[str] = None
    tensor_name: Optional[str] = None
    suggested_fix: Optional[str] = None
    explanation: Optional[str] = None
    impact: Optional[str] = None
    framework_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of UIR graph validation."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    compatibility_score: float = 1.0
    device_readiness: Dict[str, bool] = field(default_factory=dict)


class ShapeCompatibilityValidator(UIRValidator):
    """Validates tensor shape compatibility across operations."""

    def __init__(self):
        self.name = "shape_compatibility_validator"

    def validate(self, graph: UIRGraph) -> Tuple[bool, List[str]]:
        """Validate shape compatibility in the UIR graph."""
        issues = []

        try:
            # Check input-output shape compatibility for each node
            for node_id, node in graph.nodes.items():
                node_issues = self._validate_node_shapes(node, graph)
                issues.extend(node_issues)

            # Check for shape mismatches in edges
            edge_issues = self._validate_edge_shapes(graph)
            issues.extend(edge_issues)

            return len(issues) == 0, issues

        except Exception as e:
            logger.error(f"Shape validation failed: {e}")
            return False, [f"Shape validation error: {str(e)}"]

    def _validate_node_shapes(self, node: UIRNode, graph: UIRGraph) -> List[str]:
        """Validate shapes for a specific node."""
        issues = []

        # Get input and output tensors for this node
        input_tensors = [
            graph.tensors.get(tensor_name)
            for tensor_name in node.inputs
            if tensor_name in graph.tensors
        ]
        output_tensors = [
            graph.tensors.get(tensor_name)
            for tensor_name in node.outputs
            if tensor_name in graph.tensors
        ]

        # Validate based on operation type
        if node.operation_type == OperationType.CONV2D:
            issues.extend(
                self._validate_conv2d_shapes(node, input_tensors, output_tensors)
            )
        elif node.operation_type == OperationType.DENSE:
            issues.extend(
                self._validate_dense_shapes(node, input_tensors, output_tensors)
            )
        elif node.operation_type == OperationType.MAX_POOL:
            issues.extend(
                self._validate_pooling_shapes(node, input_tensors, output_tensors)
            )
        elif node.operation_type == OperationType.ADD:
            issues.extend(
                self._validate_elementwise_shapes(node, input_tensors, output_tensors)
            )
        elif node.operation_type == OperationType.CONCAT:
            issues.extend(
                self._validate_concat_shapes(node, input_tensors, output_tensors)
            )

        return issues

    def _validate_conv2d_shapes(
        self,
        node: UIRNode,
        input_tensors: List[TensorInfo],
        output_tensors: List[TensorInfo],
    ) -> List[str]:
        """Validate Conv2D operation shapes."""
        issues = []

        if len(input_tensors) < 1:
            issues.append(
                f"Conv2D node {node.node_id} requires at least one input tensor"
            )
            return issues

        input_tensor = input_tensors[0]
        if input_tensor.shape.rank < 3:
            issues.append(
                f"Conv2D node {node.node_id} requires input with at least 3 dimensions, got {input_tensor.shape.rank}"
            )

        # Check if input has proper spatial dimensions
        if input_tensor.shape.rank >= 3:
            spatial_dims = input_tensor.shape.dimensions[-2:]
            if any(dim == 0 for dim in spatial_dims if isinstance(dim, int)):
                issues.append(
                    f"Conv2D node {node.node_id} has zero spatial dimensions: {spatial_dims}"
                )

        return issues

    def _validate_dense_shapes(
        self,
        node: UIRNode,
        input_tensors: List[TensorInfo],
        output_tensors: List[TensorInfo],
    ) -> List[str]:
        """Validate Dense operation shapes."""
        issues = []

        if len(input_tensors) < 1:
            issues.append(
                f"Dense node {node.node_id} requires at least one input tensor"
            )
            return issues

        input_tensor = input_tensors[0]
        if input_tensor.shape.rank < 1:
            issues.append(
                f"Dense node {node.node_id} requires input with at least 1 dimension, got {input_tensor.shape.rank}"
            )

        return issues

    def _validate_pooling_shapes(
        self,
        node: UIRNode,
        input_tensors: List[TensorInfo],
        output_tensors: List[TensorInfo],
    ) -> List[str]:
        """Validate pooling operation shapes."""
        issues = []

        if len(input_tensors) < 1:
            issues.append(
                f"Pooling node {node.node_id} requires at least one input tensor"
            )
            return issues

        input_tensor = input_tensors[0]
        if input_tensor.shape.rank < 3:
            issues.append(
                f"Pooling node {node.node_id} requires input with at least 3 dimensions, got {input_tensor.shape.rank}"
            )

        return issues

    def _validate_elementwise_shapes(
        self,
        node: UIRNode,
        input_tensors: List[TensorInfo],
        output_tensors: List[TensorInfo],
    ) -> List[str]:
        """Validate element-wise operation shapes."""
        issues = []

        if len(input_tensors) < 2:
            issues.append(
                f"Element-wise node {node.node_id} requires at least two input tensors"
            )
            return issues

        # Check if shapes are compatible for broadcasting
        shapes = [tensor.shape for tensor in input_tensors]
        if not self._are_shapes_broadcastable(shapes):
            issues.append(
                f"Element-wise node {node.node_id} has incompatible input shapes: {[str(s) for s in shapes]}"
            )

        return issues

    def _validate_concat_shapes(
        self,
        node: UIRNode,
        input_tensors: List[TensorInfo],
        output_tensors: List[TensorInfo],
    ) -> List[str]:
        """Validate concatenation operation shapes."""
        issues = []

        if len(input_tensors) < 2:
            issues.append(
                f"Concat node {node.node_id} requires at least two input tensors"
            )
            return issues

        # Check if all tensors have the same rank
        ranks = [tensor.shape.rank for tensor in input_tensors]
        if len(set(ranks)) > 1:
            issues.append(
                f"Concat node {node.node_id} requires all inputs to have the same rank, got: {ranks}"
            )

        return issues

    def _validate_edge_shapes(self, graph: UIRGraph) -> List[str]:
        """Validate shape compatibility across edges."""
        issues = []

        for from_node_id, to_node_id, tensor_name in graph.edges:
            if tensor_name not in graph.tensors:
                issues.append(f"Edge references non-existent tensor: {tensor_name}")
                continue

            tensor = graph.tensors[tensor_name]

            # Check for zero or negative dimensions
            for i, dim in enumerate(tensor.shape.dimensions):
                if isinstance(dim, int) and dim <= 0:
                    issues.append(
                        f"Tensor {tensor_name} has invalid dimension {i}: {dim}"
                    )

        return issues

    def _are_shapes_broadcastable(self, shapes: List[TensorShape]) -> bool:
        """Check if shapes are compatible for broadcasting."""
        if len(shapes) < 2:
            return True

        # Find the maximum rank
        max_rank = max(shape.rank for shape in shapes)

        # Check each dimension from right to left
        for i in range(max_rank):
            dims = []
            for shape in shapes:
                if i < shape.rank:
                    dim = shape.dimensions[-(i + 1)]
                    dims.append(dim)

            # All dimensions must be compatible (equal, 1, or dynamic)
            if dims:
                # Filter out dynamic dimensions for comparison
                static_dims = [d for d in dims if isinstance(d, int) and d != -1]
                if static_dims:
                    # All static dimensions must be equal or 1
                    unique_dims = set(static_dims)
                    if len(unique_dims) > 1 and 1 not in unique_dims:
                        return False

        return True

    def get_name(self) -> str:
        return self.name


class OperatorSupportValidator(UIRValidator):
    """Validates operator support for target devices."""

    def __init__(self, device_capabilities: Optional[Dict[str, Dict[str, Any]]] = None):
        self.name = "operator_support_validator"
        self.device_capabilities = (
            device_capabilities or self._get_default_device_capabilities()
        )

    def _get_default_device_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get default device capabilities."""
        return {
            "raspberry_pi": {
                "supported_ops": {
                    OperationType.CONV2D,
                    OperationType.DEPTHWISE_CONV2D,
                    OperationType.MAX_POOL,
                    OperationType.AVG_POOL,
                    OperationType.RELU,
                    OperationType.SIGMOID,
                    OperationType.TANH,
                    OperationType.BATCH_NORM,
                    OperationType.DENSE,
                    OperationType.RESHAPE,
                    OperationType.FLATTEN,
                    OperationType.ADD,
                    OperationType.MUL,
                    OperationType.DIV,
                    OperationType.REDUCE_SUM,
                    OperationType.REDUCE_MEAN,
                },
                "unsupported_ops": {
                    OperationType.LSTM,
                    OperationType.GRU,
                    OperationType.RNN,
                    OperationType.ATTENTION,
                    OperationType.MULTI_HEAD_ATTENTION,
                    OperationType.GELU,
                    OperationType.SWISH,
                },
                "max_input_size": (224, 224, 3),
                "max_batch_size": 8,
            },
            "jetson_nano": {
                "supported_ops": {
                    OperationType.CONV2D,
                    OperationType.DEPTHWISE_CONV2D,
                    OperationType.MAX_POOL,
                    OperationType.AVG_POOL,
                    OperationType.RELU,
                    OperationType.SIGMOID,
                    OperationType.TANH,
                    OperationType.BATCH_NORM,
                    OperationType.DENSE,
                    OperationType.RESHAPE,
                    OperationType.FLATTEN,
                    OperationType.ADD,
                    OperationType.MUL,
                    OperationType.DIV,
                    OperationType.REDUCE_SUM,
                    OperationType.REDUCE_MEAN,
                    OperationType.LSTM,
                    OperationType.GRU,
                },
                "unsupported_ops": {
                    OperationType.ATTENTION,
                    OperationType.MULTI_HEAD_ATTENTION,
                },
                "max_input_size": (512, 512, 3),
                "max_batch_size": 16,
            },
            "cortex_m4": {
                "supported_ops": {
                    OperationType.CONV2D,
                    OperationType.DEPTHWISE_CONV2D,
                    OperationType.MAX_POOL,
                    OperationType.AVG_POOL,
                    OperationType.RELU,
                    OperationType.SIGMOID,
                    OperationType.BATCH_NORM,
                    OperationType.DENSE,
                    OperationType.RESHAPE,
                    OperationType.FLATTEN,
                    OperationType.ADD,
                    OperationType.MUL,
                    OperationType.REDUCE_SUM,
                    OperationType.REDUCE_MEAN,
                },
                "unsupported_ops": {
                    OperationType.LSTM,
                    OperationType.GRU,
                    OperationType.RNN,
                    OperationType.ATTENTION,
                    OperationType.MULTI_HEAD_ATTENTION,
                    OperationType.TANH,
                    OperationType.GELU,
                    OperationType.SWISH,
                    OperationType.DIV,
                    OperationType.POW,
                    OperationType.SQRT,
                },
                "max_input_size": (112, 112, 3),
                "max_batch_size": 1,
            },
            "cpu": {
                "supported_ops": set(OperationType),  # CPU supports all operations
                "unsupported_ops": set(),
                "max_input_size": (1024, 1024, 3),
                "max_batch_size": 64,
            },
            "gpu": {
                "supported_ops": set(OperationType),  # GPU supports all operations
                "unsupported_ops": set(),
                "max_input_size": (2048, 2048, 3),
                "max_batch_size": 128,
            },
        }

    def validate(self, graph: UIRGraph) -> Tuple[bool, List[str]]:
        """Validate operator support for target devices."""
        issues = []

        try:
            # Get target device from graph metadata
            target_device = graph.framework_metadata.get("target_device", "cpu")

            if target_device not in self.device_capabilities:
                issues.append(f"Unknown target device: {target_device}")
                return False, issues

            device_caps = self.device_capabilities[target_device]
            supported_ops = device_caps["supported_ops"]
            unsupported_ops = device_caps["unsupported_ops"]

            # Check each node for operator support
            for node_id, node in graph.nodes.items():
                if node.operation_type in unsupported_ops:
                    issues.append(
                        f"Node {node_id} uses unsupported operation {node.operation_type.value} "
                        f"for device {target_device}"
                    )
                elif node.operation_type not in supported_ops:
                    issues.append(
                        f"Node {node_id} uses operation {node.operation_type.value} "
                        f"with unknown support status for device {target_device}"
                    )

            # Check input size constraints
            input_size_issues = self._validate_input_size_constraints(
                graph, device_caps
            )
            issues.extend(input_size_issues)

            return len(issues) == 0, issues

        except Exception as e:
            logger.error(f"Operator support validation failed: {e}")
            return False, [f"Operator support validation error: {str(e)}"]

    def _validate_input_size_constraints(
        self, graph: UIRGraph, device_caps: Dict[str, Any]
    ) -> List[str]:
        """Validate input size constraints for the target device."""
        issues = []

        max_input_size = device_caps.get("max_input_size")
        max_batch_size = device_caps.get("max_batch_size")

        if not max_input_size and not max_batch_size:
            return issues

        # Find input tensors
        input_tensors = [
            tensor
            for tensor in graph.tensors.values()
            if tensor.framework_metadata.get("input", False) or tensor.name == "input"
        ]

        for tensor in input_tensors:
            if tensor.shape.rank >= 3:  # Assume spatial dimensions
                # Check batch size
                if max_batch_size and tensor.shape.dimensions[0] > max_batch_size:
                    issues.append(
                        f"Input tensor {tensor.name} batch size {tensor.shape.dimensions[0]} "
                        f"exceeds device limit {max_batch_size}"
                    )

                # Check spatial dimensions
                if max_input_size and len(max_input_size) >= 2:
                    spatial_dims = tensor.shape.dimensions[-2:]
                    if (
                        isinstance(spatial_dims[0], int)
                        and spatial_dims[0] > max_input_size[0]
                    ) or (
                        isinstance(spatial_dims[1], int)
                        and spatial_dims[1] > max_input_size[1]
                    ):
                        issues.append(
                            f"Input tensor {tensor.name} spatial dimensions {spatial_dims} "
                            f"exceed device limit {max_input_size[:2]}"
                        )

        return issues

    def get_name(self) -> str:
        return self.name


class DataTypeCompatibilityValidator(UIRValidator):
    """Validates data type compatibility and quantization support."""

    def __init__(self):
        self.name = "data_type_compatibility_validator"

    def validate(self, graph: UIRGraph) -> Tuple[bool, List[str]]:
        """Validate data type compatibility in the UIR graph."""
        issues = []

        try:
            # Check for unsupported data types
            unsupported_dtypes = self._get_unsupported_data_types(graph)
            issues.extend(unsupported_dtypes)

            # Check for data type mismatches in operations
            dtype_mismatches = self._check_data_type_mismatches(graph)
            issues.extend(dtype_mismatches)

            # Check quantization compatibility
            quantization_issues = self._check_quantization_compatibility(graph)
            issues.extend(quantization_issues)

            return len(issues) == 0, issues

        except Exception as e:
            logger.error(f"Data type validation failed: {e}")
            return False, [f"Data type validation error: {str(e)}"]

    def _get_unsupported_data_types(self, graph: UIRGraph) -> List[str]:
        """Check for unsupported data types."""
        issues = []

        # Define unsupported data types for edge devices
        unsupported_dtypes = {
            DataType.COMPLEX64,
            DataType.COMPLEX128,
            DataType.FLOAT64,  # Often not supported on edge devices
        }

        for tensor_name, tensor in graph.tensors.items():
            if tensor.dtype in unsupported_dtypes:
                issues.append(
                    f"Tensor {tensor_name} uses unsupported data type {tensor.dtype.value} "
                    f"for edge deployment"
                )

        return issues

    def _check_data_type_mismatches(self, graph: UIRGraph) -> List[str]:
        """Check for data type mismatches in operations."""
        issues = []

        for node_id, node in graph.nodes.items():
            if node.operation_type in [
                OperationType.ADD,
                OperationType.SUB,
                OperationType.MUL,
                OperationType.DIV,
            ]:
                # Element-wise operations require compatible data types
                input_tensors = [
                    graph.tensors.get(tensor_name)
                    for tensor_name in node.inputs
                    if tensor_name in graph.tensors
                ]

                if len(input_tensors) >= 2:
                    dtypes = [tensor.dtype for tensor in input_tensors]
                    if len(set(dtypes)) > 1:
                        issues.append(
                            f"Node {node_id} has input tensors with incompatible data types: {[d.value for d in dtypes]}"
                        )

        return issues

    def _check_quantization_compatibility(self, graph: UIRGraph) -> List[str]:
        """Check quantization compatibility."""
        issues = []

        # Get quantization settings from graph metadata
        quantize_type = graph.framework_metadata.get("quantize", "none")

        if quantize_type == "int8":
            # Check for operations that don't support INT8 quantization
            unsupported_int8_ops = {
                OperationType.SOFTMAX,
                OperationType.SIGMOID,
                OperationType.TANH,
                OperationType.GELU,
                OperationType.SWISH,
            }

            for node_id, node in graph.nodes.items():
                if node.operation_type in unsupported_int8_ops:
                    issues.append(
                        f"Node {node_id} with operation {node.operation_type.value} "
                        f"may not support INT8 quantization"
                    )

        return issues

    def get_name(self) -> str:
        return self.name


class PerformanceValidator(UIRValidator):
    """Validates performance characteristics and optimization opportunities."""

    def __init__(self):
        self.name = "performance_validator"

    def validate(self, graph: UIRGraph) -> Tuple[bool, List[str]]:
        """Validate performance characteristics."""
        issues = []

        try:
            # Check for performance bottlenecks
            bottleneck_issues = self._check_performance_bottlenecks(graph)
            issues.extend(bottleneck_issues)

            # Check for optimization opportunities
            optimization_issues = self._check_optimization_opportunities(graph)
            issues.extend(optimization_issues)

            # Check memory usage patterns
            memory_issues = self._check_memory_patterns(graph)
            issues.extend(memory_issues)

            return len(issues) == 0, issues

        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return False, [f"Performance validation error: {str(e)}"]

    def _check_performance_bottlenecks(self, graph: UIRGraph) -> List[str]:
        """Check for potential performance bottlenecks."""
        issues = []

        # Check for large tensor operations
        for tensor_name, tensor in graph.tensors.items():
            if tensor.shape.is_dynamic():
                continue

            # Calculate tensor size
            total_elements = 1
            for dim in tensor.shape.dimensions:
                if isinstance(dim, int):
                    total_elements *= dim

            # Flag large tensors
            if total_elements > 10_000_000:  # 10M elements
                issues.append(
                    f"Large tensor {tensor_name} with {total_elements:,} elements "
                    f"may cause performance issues on edge devices"
                )

        return issues

    def _check_optimization_opportunities(self, graph: UIRGraph) -> List[str]:
        """Check for optimization opportunities."""
        issues = []

        # Check for consecutive operations that could be fused
        fusion_opportunities = self._find_fusion_opportunities(graph)
        issues.extend(fusion_opportunities)

        # Check for redundant operations
        redundant_ops = self._find_redundant_operations(graph)
        issues.extend(redundant_ops)

        return issues

    def _find_fusion_opportunities(self, graph: UIRGraph) -> List[str]:
        """Find opportunities for operation fusion."""
        issues = []

        # Look for Conv2D + BatchNorm + ReLU patterns
        for node_id, node in graph.nodes.items():
            if node.operation_type == OperationType.CONV2D:
                # Check if this Conv2D is followed by BatchNorm and ReLU
                output_tensors = [
                    tensor_name
                    for tensor_name in node.outputs
                    if tensor_name in graph.tensors
                ]

                for output_tensor in output_tensors:
                    # Find nodes that consume this tensor
                    consumer_nodes = [
                        edge[1]
                        for edge in graph.edges
                        if edge[0] == node_id and edge[2] == output_tensor
                    ]

                    if len(consumer_nodes) == 1:
                        consumer_node = graph.nodes.get(consumer_nodes[0])
                        if (
                            consumer_node
                            and consumer_node.operation_type == OperationType.BATCH_NORM
                        ):
                            # Check for ReLU after BatchNorm
                            bn_outputs = [
                                tensor_name
                                for tensor_name in consumer_node.outputs
                                if tensor_name in graph.tensors
                            ]
                            for bn_output in bn_outputs:
                                bn_consumers = [
                                    edge[1]
                                    for edge in graph.edges
                                    if edge[0] == consumer_nodes[0]
                                    and edge[2] == bn_output
                                ]

                                for bn_consumer in bn_consumers:
                                    bn_consumer_node = graph.nodes.get(bn_consumer)
                                    if (
                                        bn_consumer_node
                                        and bn_consumer_node.operation_type
                                        == OperationType.RELU
                                    ):
                                        issues.append(
                                            f"Fusion opportunity: Conv2D ({node_id}) + BatchNorm ({consumer_nodes[0]}) + ReLU ({bn_consumer}) "
                                            f"can be fused for better performance"
                                        )

        return issues

    def _find_redundant_operations(self, graph: UIRGraph) -> List[str]:
        """Find redundant operations."""
        issues = []

        # Check for identity operations
        for node_id, node in graph.nodes.items():
            if node.operation_type == OperationType.RESHAPE:
                # Check if reshape is actually changing the shape
                input_tensors = [
                    graph.tensors.get(tensor_name)
                    for tensor_name in node.inputs
                    if tensor_name in graph.tensors
                ]
                output_tensors = [
                    graph.tensors.get(tensor_name)
                    for tensor_name in node.outputs
                    if tensor_name in graph.tensors
                ]

                if input_tensors and output_tensors:
                    input_shape = input_tensors[0].shape
                    output_shape = output_tensors[0].shape

                    if input_shape.dimensions == output_shape.dimensions:
                        issues.append(
                            f"Redundant reshape operation {node_id}: input and output shapes are identical"
                        )

        return issues

    def _check_memory_patterns(self, graph: UIRGraph) -> List[str]:
        """Check memory usage patterns."""
        issues = []

        # Check for high memory operations
        high_memory_ops = {
            OperationType.CONCAT,
            OperationType.SPLIT,
            OperationType.STACK,
        }

        for node_id, node in graph.nodes.items():
            if node.operation_type in high_memory_ops:
                issues.append(
                    f"High memory operation {node.operation_type.value} at node {node_id} "
                    f"may cause memory pressure on edge devices"
                )

        return issues

    def get_name(self) -> str:
        return self.name


class UIRValidationSuite:
    """Comprehensive validation suite for UIR graphs."""

    def __init__(self):
        self.validators: List[UIRValidator] = [
            ShapeCompatibilityValidator(),
            OperatorSupportValidator(),
            DataTypeCompatibilityValidator(),
            PerformanceValidator(),
        ]

    def validate_graph(self, graph: UIRGraph) -> ValidationResult:
        """Run comprehensive validation on a UIR graph."""
        result = ValidationResult(is_valid=True)

        for validator in self.validators:
            try:
                is_valid, issues = validator.validate(graph)

                if not is_valid:
                    result.is_valid = False

                # Convert string issues to ValidationIssue objects
                for issue in issues:
                    validation_issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.SEMANTIC,
                        message=issue,
                        framework_metadata={"validator": validator.get_name()},
                    )
                    result.issues.append(validation_issue)

            except Exception as e:
                logger.error(f"Validator {validator.get_name()} failed: {e}")
                result.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.SEMANTIC,
                        message=f"Validator {validator.get_name()} failed: {str(e)}",
                        framework_metadata={
                            "validator": validator.get_name(),
                            "error": str(e),
                        },
                    )
                )
                result.is_valid = False

        # Calculate compatibility score
        result.compatibility_score = self._calculate_compatibility_score(result)

        # Check device readiness
        result.device_readiness = self._check_device_readiness(graph, result)

        return result

    def _calculate_compatibility_score(self, result: ValidationResult) -> float:
        """Calculate overall compatibility score."""
        total_issues = len(result.issues) + len(result.warnings)
        if total_issues == 0:
            return 1.0

        # Weight errors more heavily than warnings
        error_weight = 0.8
        warning_weight = 0.2

        error_score = len(result.issues) * error_weight
        warning_score = len(result.warnings) * warning_weight

        total_score = error_score + warning_score
        return max(0.0, 1.0 - (total_score / 10.0))

    def _check_device_readiness(
        self, graph: UIRGraph, result: ValidationResult
    ) -> Dict[str, bool]:
        """Check readiness for different target devices."""
        devices = ["raspberry_pi", "jetson_nano", "cortex_m4", "cpu", "gpu"]
        readiness = {}

        for device in devices:
            # Check if there are any device-specific errors
            device_errors = [
                issue for issue in result.issues if device in issue.message.lower()
            ]
            readiness[device] = len(device_errors) == 0

        return readiness


def validate_uir_graph(graph: UIRGraph) -> ValidationResult:
    """Main function for validating UIR graphs.

    Args:
        graph: UIR graph to validate

    Returns:
        ValidationResult with detailed validation information
    """
    suite = UIRValidationSuite()
    return suite.validate_graph(graph)


if __name__ == "__main__":
    # Test the validators
    from unified_ir import create_uir_from_edgeflow_config

    test_config = {
        "model": "test_model.tflite",
        "quantize": "int8",
        "target_device": "raspberry_pi",
        "input_shape": "1,224,224,3",
    }

    graph = create_uir_from_edgeflow_config(test_config)
    result = validate_uir_graph(graph)

    print(f"Validation Result:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Compatibility Score: {result.compatibility_score:.2f}")
    print(f"  Issues: {len(result.issues)}")
    print(f"  Warnings: {len(result.warnings)}")
    print(f"  Device Readiness: {result.device_readiness}")

    if result.issues:
        print("\nIssues:")
        for issue in result.issues:
            print(f"  - {issue.message}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning.message}")
