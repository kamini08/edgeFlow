"""
Main semantic analyzer that performs comprehensive validation of DSL IR graphs.
"""
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from .constraints import ConstraintConfig, DeviceConstraints, ParameterRange
from .error_types import (
    ErrorCollector,
    ErrorSeverity,
    ErrorType,
    SemanticError,
    SourceLocation,
)
from .ir_nodes import (
    ActivationType,
    DataType,
    IRGraph,
    IRNode,
    LayerType,
    TensorInfo,
    TensorShape,
)


class SemanticAnalyzer:
    """Main semantic analyzer for DSL IR graphs."""

    def __init__(self, config: Optional[ConstraintConfig] = None):
        self.config = config or ConstraintConfig()
        self.error_collector = ErrorCollector()
        self.visited_nodes: Set[str] = set()
        self.analysis_context: Dict[str, Any] = {}

    def analyze(self, graph: IRGraph) -> ErrorCollector:
        """Perform complete semantic analysis on the IR graph."""
        self.error_collector = ErrorCollector()  # Reset errors
        self.visited_nodes.clear()
        self.analysis_context.clear()

        # Phase 1: Basic graph structure validation
        self._validate_graph_structure(graph)

        # Phase 2: Node-level validation
        self._validate_nodes(graph)

        # Phase 3: Edge/connection validation
        self._validate_connections(graph)

        # Phase 4: Parameter validation
        self._validate_parameters(graph)

        # Phase 5: Layer compatibility validation
        self._validate_layer_compatibility(graph)

        # Phase 6: Sequence and configuration validation
        self._validate_sequences_and_configurations(graph)

        # Phase 7: Resource constraint validation
        self._validate_resource_constraints(graph)

        # Phase 8: Device compatibility validation
        self._validate_device_compatibility(graph)

        return self.error_collector

    def _validate_graph_structure(self, graph: IRGraph) -> None:
        """Validate basic graph structure."""
        # Check for empty graph
        if len(graph) == 0:
            self.error_collector.add_error(
                SemanticError(
                    error_type=ErrorType.GRAPH_CYCLE,
                    severity=ErrorSeverity.ERROR,
                    message="Graph is empty",
                )
            )
            return

        # Check for cycles
        if graph.has_cycles():
            self.error_collector.add_error(
                SemanticError(
                    error_type=ErrorType.GRAPH_CYCLE,
                    severity=ErrorSeverity.ERROR,
                    message="Graph contains cycles, which are not allowed",
                )
            )

        # Check connectivity
        if not graph.is_connected():
            self.error_collector.add_error(
                SemanticError(
                    error_type=ErrorType.CONNECTIVITY,
                    severity=ErrorSeverity.WARNING,
                    message="Graph has disconnected components",
                )
            )

        # Check for input and output nodes
        if not graph.input_nodes:
            self.error_collector.add_error(
                SemanticError(
                    error_type=ErrorType.MISSING_LAYER,
                    severity=ErrorSeverity.ERROR,
                    message="Graph must have at least one Input layer",
                )
            )

        if not graph.output_nodes:
            self.error_collector.add_error(
                SemanticError(
                    error_type=ErrorType.MISSING_LAYER,
                    severity=ErrorSeverity.WARNING,
                    message="Graph should have at least one Output layer",
                )
            )

    def _validate_nodes(self, graph: IRGraph) -> None:
        """Validate individual nodes."""
        for node in graph.nodes.values():
            self._validate_single_node(node)

    def _validate_single_node(self, node: IRNode) -> None:
        """Validate a single node."""
        # Check required parameters exist
        required_params = self._get_required_parameters(node.layer_type)
        for param_name in required_params:
            if param_name not in node.parameters:
                self.error_collector.add_error(
                    SemanticError(
                        error_type=ErrorType.PARAM_RANGE,
                        severity=ErrorSeverity.ERROR,
                        message=f"Missing required parameter '{param_name}'",
                        node_id=node.node_id,
                        layer_name=node.display_name,
                        location=node.location,
                    )
                )

        # Validate tensor shapes are properly defined
        for i, tensor in enumerate(node.input_tensors):
            if not tensor.shape.dimensions:
                self.error_collector.add_error(
                    SemanticError(
                        error_type=ErrorType.SHAPE_MISMATCH,
                        severity=ErrorSeverity.ERROR,
                        message=f"Input tensor {i} has undefined shape",
                        node_id=node.node_id,
                        layer_name=node.display_name,
                        location=node.location,
                    )
                )

        for i, tensor in enumerate(node.output_tensors):
            if not tensor.shape.dimensions:
                self.error_collector.add_error(
                    SemanticError(
                        error_type=ErrorType.SHAPE_MISMATCH,
                        severity=ErrorSeverity.ERROR,
                        message=f"Output tensor {i} has undefined shape",
                        node_id=node.node_id,
                        layer_name=node.display_name,
                        location=node.location,
                    )
                )

    def _validate_connections(self, graph: IRGraph) -> None:
        """Validate connections between nodes."""
        for node in graph.nodes.values():
            # Check input connections
            for input_node_id in node.input_nodes:
                input_node = graph.get_node(input_node_id)
                if not input_node:
                    self.error_collector.add_error(
                        SemanticError(
                            error_type=ErrorType.CONNECTIVITY,
                            severity=ErrorSeverity.ERROR,
                            message=f"Referenced input node '{input_node_id}' does not exist",
                            node_id=node.node_id,
                            layer_name=node.display_name,
                            location=node.location,
                        )
                    )
                    continue

                # Validate tensor compatibility
                self._validate_tensor_compatibility(input_node, node)

            # Check output connections
            for output_node_id in node.output_nodes:
                output_node = graph.get_node(output_node_id)
                if not output_node:
                    self.error_collector.add_error(
                        SemanticError(
                            error_type=ErrorType.CONNECTIVITY,
                            severity=ErrorSeverity.ERROR,
                            message=f"Referenced output node '{output_node_id}' does not exist",
                            node_id=node.node_id,
                            layer_name=node.display_name,
                            location=node.location,
                        )
                    )

    def _validate_tensor_compatibility(
        self, producer: IRNode, consumer: IRNode
    ) -> None:
        """Validate tensor compatibility between connected nodes."""
        if not producer.output_tensors or not consumer.input_tensors:
            return  # Skip if tensors are not defined

        # For now, assume single input/output per node (can be extended)
        producer_output = producer.output_tensors[0]
        consumer_input = consumer.input_tensors[0]

        # Check shape compatibility
        if not producer_output.shape.is_compatible_with(consumer_input.shape):
            self.error_collector.add_shape_mismatch(
                expected_shape=consumer_input.shape.dimensions,
                actual_shape=producer_output.shape.dimensions,
                node_id=consumer.node_id,
                layer_name=consumer.display_name,
                location=consumer.location,
            )

        # Check datatype compatibility
        if producer_output.dtype != consumer_input.dtype:
            self.error_collector.add_error(
                SemanticError(
                    error_type=ErrorType.DATATYPE_MISMATCH,
                    severity=ErrorSeverity.WARNING,
                    message=f"Datatype mismatch: producer outputs {producer_output.dtype.value}, "
                    f"consumer expects {consumer_input.dtype.value}",
                    node_id=consumer.node_id,
                    layer_name=consumer.display_name,
                    location=consumer.location,
                    context={
                        "producer_dtype": producer_output.dtype.value,
                        "consumer_dtype": consumer_input.dtype.value,
                    },
                )
            )

    def _validate_parameters(self, graph: IRGraph) -> None:
        """Validate parameters for all nodes."""
        for node in graph.nodes.values():
            layer_params = self.config.parameter_ranges.get(node.layer_type, {})

            for param_name, param_value in node.parameters.items():
                param_range = layer_params.get(param_name)
                if param_range and not param_range.is_valid(param_value):
                    if param_range.valid_values:
                        self.error_collector.add_error(
                            SemanticError(
                                error_type=ErrorType.PARAM_RANGE,
                                severity=ErrorSeverity.ERROR,
                                message=f"Parameter '{param_name}' value {param_value} not in valid values {param_range.valid_values}",
                                node_id=node.node_id,
                                layer_name=node.display_name,
                                location=node.location,
                            )
                        )
                    else:
                        self.error_collector.add_param_range_error(
                            param_name=param_name,
                            value=param_value,
                            min_val=param_range.min_value,
                            max_val=param_range.max_value,
                            node_id=node.node_id,
                            layer_name=node.display_name,
                            location=node.location,
                        )

    def _validate_layer_compatibility(self, graph: IRGraph) -> None:
        """Validate layer-specific compatibility rules."""
        execution_order = graph.get_execution_order()

        for i, node_id in enumerate(execution_order):
            node = graph.get_node(node_id)
            if not node:
                continue

            compatibility_rule = self.config.get_compatibility_rule(node.layer_type)
            if not compatibility_rule:
                continue

            # Check input rank requirements
            if "input_rank" in compatibility_rule and node.input_tensors:
                expected_ranks = compatibility_rule["input_rank"]
                if isinstance(expected_ranks, int):
                    expected_ranks = [expected_ranks]

                actual_rank = node.input_tensors[0].shape.rank
                if actual_rank not in expected_ranks:
                    self.error_collector.add_error(
                        SemanticError(
                            error_type=ErrorType.SHAPE_MISMATCH,
                            severity=ErrorSeverity.ERROR,
                            message=f"Layer expects input rank {expected_ranks}, got {actual_rank}",
                            node_id=node.node_id,
                            layer_name=node.display_name,
                            location=node.location,
                        )
                    )

            # Check if flatten is required before dense layers
            if node.layer_type == LayerType.DENSE and compatibility_rule.get(
                "requires_flatten", False
            ):
                predecessors = graph.get_predecessors(node_id)
                if predecessors:
                    prev_node = predecessors[0]
                    if prev_node.layer_type in [
                        LayerType.CONV2D,
                        LayerType.MAXPOOL2D,
                        LayerType.AVGPOOL2D,
                    ] and not self._has_flatten_between(
                        graph, prev_node.node_id, node_id
                    ):
                        self.error_collector.add_error(
                            SemanticError(
                                error_type=ErrorType.INVALID_SEQUENCE,
                                severity=ErrorSeverity.ERROR,
                                message=f"Dense layer requires Flatten layer after {prev_node.layer_type.value}",
                                node_id=node.node_id,
                                layer_name=node.display_name,
                                location=node.location,
                                suggestion="Add a Flatten layer between the convolutional and dense layers",
                            )
                        )

    def _validate_sequences_and_configurations(self, graph: IRGraph) -> None:
        """Validate forbidden sequences and configurations."""
        # Check forbidden sequences
        execution_order = graph.get_execution_order()
        for i in range(len(execution_order) - 1):
            curr_node = graph.get_node(execution_order[i])
            next_node = graph.get_node(execution_order[i + 1])

            if curr_node and next_node:
                if self.config.is_sequence_forbidden(
                    curr_node.layer_type, next_node.layer_type
                ):
                    self.error_collector.add_forbidden_config(
                        config_description=f"{curr_node.layer_type.value} followed by {next_node.layer_type.value}",
                        node_id=next_node.node_id,
                        layer_name=next_node.display_name,
                        location=next_node.location,
                        suggestion=f"Add appropriate intermediate layers between {curr_node.layer_type.value} and {next_node.layer_type.value}",
                    )

        # Check forbidden combinations
        for forbidden_combo in self.config.forbidden_combinations:
            if forbidden_combo["condition"](graph):
                self.error_collector.add_forbidden_config(
                    config_description=forbidden_combo["description"],
                    suggestion="Review the model architecture and pipeline configuration",
                )

    def _validate_resource_constraints(self, graph: IRGraph) -> None:
        """Validate resource usage against device constraints."""
        device_limits = self.config.device_constraints

        # Calculate total memory usage
        total_memory_bytes = graph.calculate_total_memory_usage()
        total_memory_mb = total_memory_bytes / (1024 * 1024)

        if total_memory_mb > device_limits.max_memory_mb:
            self.error_collector.add_resource_limit_error(
                resource_type="Memory",
                usage=total_memory_mb,
                limit=device_limits.max_memory_mb,
            )

        # Check individual tensor sizes
        for node in graph.nodes.values():
            for tensor in node.input_tensors + node.output_tensors:
                if tensor.shape.total_elements > device_limits.max_tensor_size:
                    self.error_collector.add_resource_limit_error(
                        resource_type="Tensor size",
                        usage=tensor.shape.total_elements,
                        limit=device_limits.max_tensor_size,
                        node_id=node.node_id,
                        layer_name=node.display_name,
                        location=node.location,
                    )

        # Check layer-specific constraints
        for node in graph.nodes.values():
            if node.layer_type == LayerType.CONV2D:
                filters = node.get_param("filters", 0)
                if filters > device_limits.max_filters:
                    self.error_collector.add_resource_limit_error(
                        resource_type="Conv2D filters",
                        usage=filters,
                        limit=device_limits.max_filters,
                        node_id=node.node_id,
                        layer_name=node.display_name,
                        location=node.location,
                    )

                kernel_size = node.get_param("kernel_size", (1, 1))
                max_kernel = (
                    max(kernel_size)
                    if isinstance(kernel_size, (list, tuple))
                    else kernel_size
                )
                if max_kernel > device_limits.max_kernel_size:
                    self.error_collector.add_resource_limit_error(
                        resource_type="Kernel size",
                        usage=max_kernel,
                        limit=device_limits.max_kernel_size,
                        node_id=node.node_id,
                        layer_name=node.display_name,
                        location=node.location,
                    )

            elif node.layer_type == LayerType.DENSE:
                units = node.get_param("units", 0)
                if units > device_limits.max_units:
                    self.error_collector.add_resource_limit_error(
                        resource_type="Dense units",
                        usage=units,
                        limit=device_limits.max_units,
                        node_id=node.node_id,
                        layer_name=node.display_name,
                        location=node.location,
                    )

    def _validate_device_compatibility(self, graph: IRGraph) -> None:
        """Validate compatibility with target device."""
        device_constraints = self.config.device_constraints

        # Check supported layer types
        for node in graph.nodes.values():
            if node.layer_type not in device_constraints.supported_layers:
                self.error_collector.add_error(
                    SemanticError(
                        error_type=ErrorType.UNSUPPORTED_OP,
                        severity=ErrorSeverity.ERROR,
                        message=f"Layer type {node.layer_type.value} is not supported on target device",
                        node_id=node.node_id,
                        layer_name=node.display_name,
                        location=node.location,
                    )
                )

        # Check supported data types
        for node in graph.nodes.values():
            for tensor in node.input_tensors + node.output_tensors:
                if tensor.dtype not in device_constraints.supported_dtypes:
                    self.error_collector.add_error(
                        SemanticError(
                            error_type=ErrorType.DEVICE_INCOMPATIBLE,
                            severity=ErrorSeverity.ERROR,
                            message=f"Data type {tensor.dtype.value} is not supported on target device",
                            node_id=node.node_id,
                            layer_name=node.display_name,
                            location=node.location,
                            suggestion=f"Use one of the supported data types: {[dt.value for dt in device_constraints.supported_dtypes]}",
                        )
                    )

    def _get_required_parameters(self, layer_type: LayerType) -> List[str]:
        """Get list of required parameters for a layer type."""
        required_params = {
            LayerType.DENSE: ["units"],
            LayerType.CONV2D: ["filters", "kernel_size"],
            LayerType.CONV1D: ["filters", "kernel_size"],
            LayerType.MAXPOOL2D: ["pool_size"],
            LayerType.AVGPOOL2D: ["pool_size"],
            LayerType.DROPOUT: ["rate"],
            LayerType.LSTM: ["units"],
            LayerType.GRU: ["units"],
            LayerType.EMBEDDING: ["input_dim", "output_dim"],
        }
        return required_params.get(layer_type, [])

    def _has_flatten_between(
        self, graph: IRGraph, start_node_id: str, end_node_id: str
    ) -> bool:
        """Check if there's a Flatten layer between two nodes."""
        visited = set()
        queue = [start_node_id]

        while queue:
            current_id = queue.pop(0)
            if current_id == end_node_id:
                return False  # Reached end without finding Flatten

            if current_id in visited:
                continue
            visited.add(current_id)

            current_node = graph.get_node(current_id)
            if current_node:
                if current_node.layer_type == LayerType.FLATTEN:
                    return True

                for output_id in current_node.output_nodes:
                    if output_id not in visited:
                        queue.append(output_id)

        return False

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis results."""
        return {
            "total_errors": len(self.error_collector.errors),
            "error_counts": dict(self.error_collector.error_counts),
            "has_blocking_errors": self.error_collector.has_errors(),
            "has_fatal_errors": self.error_collector.has_fatal_errors(),
            "errors_by_type": self._group_errors_by_type(),
        }

    def _group_errors_by_type(self) -> Dict[str, int]:
        """Group errors by type for summary."""
        error_types = {}
        for error in self.error_collector.errors:
            error_type = error.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1
        return error_types


def semantic_check(
    graph: IRGraph, config: Optional[ConstraintConfig] = None
) -> ErrorCollector:
    """Convenience function to perform semantic analysis."""
    analyzer = SemanticAnalyzer(config)
    return analyzer.analyze(graph)
