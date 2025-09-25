"""
Configuration system for parameter ranges, forbidden configurations, and device constraints.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .ir_nodes import ActivationType, DataType, LayerType


@dataclass
class ParameterRange:
    """Defines valid range for a parameter."""

    min_value: Union[int, float]
    max_value: Union[int, float]
    valid_values: Optional[List[Any]] = None  # For enum-like parameters
    validator: Optional[Callable[[Any], bool]] = None  # Custom validation function

    def is_valid(self, value: Any) -> bool:
        """Check if a value is within the valid range."""
        if self.valid_values is not None:
            return value in self.valid_values

        if self.validator is not None:
            return self.validator(value)

        try:
            numeric_value = float(value)
            return self.min_value <= numeric_value <= self.max_value
        except (ValueError, TypeError):
            return False


@dataclass
class DeviceConstraints:
    """Hardware device constraints and limits."""

    max_memory_mb: float = 1024.0  # Maximum memory in MB
    max_compute_ops: int = 1000000  # Maximum compute operations
    supported_dtypes: List[DataType] = field(default_factory=lambda: list(DataType))
    supported_layers: List[LayerType] = field(default_factory=lambda: list(LayerType))
    max_tensor_size: int = 1000000  # Maximum elements in a tensor
    max_kernel_size: int = 11  # Maximum kernel size for convolutions
    max_filters: int = 1024  # Maximum number of filters
    max_units: int = 4096  # Maximum units in dense layers

    def __post_init__(self):
        if not self.supported_dtypes:
            self.supported_dtypes = list(DataType)
        if not self.supported_layers:
            self.supported_layers = list(LayerType)


class ConstraintConfig:
    """Configuration for all semantic analysis constraints."""

    def __init__(self):
        self.parameter_ranges = self._init_parameter_ranges()
        self.forbidden_sequences = self._init_forbidden_sequences()
        self.forbidden_combinations = self._init_forbidden_combinations()
        self.device_constraints = DeviceConstraints()
        self.layer_compatibility_rules = self._init_compatibility_rules()

    def _init_parameter_ranges(self) -> Dict[LayerType, Dict[str, ParameterRange]]:
        """Initialize parameter ranges for each layer type."""
        return {
            LayerType.DENSE: {
                "units": ParameterRange(1, 10000),
                "activation": ParameterRange(0, 0, valid_values=list(ActivationType)),
                "use_bias": ParameterRange(0, 0, valid_values=[True, False]),
            },
            LayerType.CONV2D: {
                "filters": ParameterRange(1, 1024),
                "kernel_size": ParameterRange(
                    1, 11, validator=self._validate_kernel_size
                ),
                "strides": ParameterRange(1, 5, validator=self._validate_strides),
                "padding": ParameterRange(0, 0, valid_values=["valid", "same"]),
                "activation": ParameterRange(0, 0, valid_values=list(ActivationType)),
                "use_bias": ParameterRange(0, 0, valid_values=[True, False]),
            },
            LayerType.CONV1D: {
                "filters": ParameterRange(1, 1024),
                "kernel_size": ParameterRange(1, 21),
                "strides": ParameterRange(1, 5),
                "padding": ParameterRange(0, 0, valid_values=["valid", "same"]),
                "activation": ParameterRange(0, 0, valid_values=list(ActivationType)),
            },
            LayerType.MAXPOOL2D: {
                "pool_size": ParameterRange(1, 8, validator=self._validate_pool_size),
                "strides": ParameterRange(1, 8, validator=self._validate_strides),
                "padding": ParameterRange(0, 0, valid_values=["valid", "same"]),
            },
            LayerType.AVGPOOL2D: {
                "pool_size": ParameterRange(1, 8, validator=self._validate_pool_size),
                "strides": ParameterRange(1, 8, validator=self._validate_strides),
                "padding": ParameterRange(0, 0, valid_values=["valid", "same"]),
            },
            LayerType.DROPOUT: {"rate": ParameterRange(0.0, 0.9)},
            LayerType.BATCH_NORM: {
                "momentum": ParameterRange(0.0, 1.0),
                "epsilon": ParameterRange(1e-8, 1e-3),
            },
            LayerType.LAYER_NORM: {"epsilon": ParameterRange(1e-8, 1e-3)},
            LayerType.LSTM: {
                "units": ParameterRange(1, 2048),
                "return_sequences": ParameterRange(0, 0, valid_values=[True, False]),
                "return_state": ParameterRange(0, 0, valid_values=[True, False]),
                "dropout": ParameterRange(0.0, 0.9),
                "recurrent_dropout": ParameterRange(0.0, 0.9),
            },
            LayerType.GRU: {
                "units": ParameterRange(1, 2048),
                "return_sequences": ParameterRange(0, 0, valid_values=[True, False]),
                "return_state": ParameterRange(0, 0, valid_values=[True, False]),
                "dropout": ParameterRange(0.0, 0.9),
                "recurrent_dropout": ParameterRange(0.0, 0.9),
            },
            LayerType.EMBEDDING: {
                "input_dim": ParameterRange(1, 1000000),
                "output_dim": ParameterRange(1, 1024),
                "input_length": ParameterRange(1, 10000),
            },
        }

    def _init_forbidden_sequences(self) -> List[Tuple[LayerType, LayerType]]:
        """Initialize forbidden layer sequences."""
        return [
            # Dense layer cannot follow non-flattened conv layers directly
            (LayerType.CONV2D, LayerType.DENSE),
            (LayerType.MAXPOOL2D, LayerType.DENSE),
            (LayerType.AVGPOOL2D, LayerType.DENSE),
            # Pooling layers cannot follow 1D convolutions
            (LayerType.CONV1D, LayerType.MAXPOOL2D),
            (LayerType.CONV1D, LayerType.AVGPOOL2D),
            # Recurrent layers have specific input requirements
            (LayerType.FLATTEN, LayerType.LSTM),
            (LayerType.FLATTEN, LayerType.GRU),
        ]

    def _init_forbidden_combinations(self) -> List[Dict[str, Any]]:
        """Initialize forbidden parameter combinations."""
        return [
            {
                "description": "Pruning cannot precede quantization",
                "condition": lambda graph: self._check_pruning_before_quantization(
                    graph
                ),
            },
            {
                "description": "Dropout rate too high with batch normalization",
                "condition": lambda graph: self._check_high_dropout_with_batchnorm(
                    graph
                ),
            },
            {
                "description": "Too many consecutive dropout layers",
                "condition": lambda graph: self._check_consecutive_dropout(graph),
            },
        ]

    def _init_compatibility_rules(self) -> Dict[LayerType, Dict[str, Any]]:
        """Initialize layer compatibility rules."""
        return {
            LayerType.DENSE: {
                "requires_flatten": True,  # Needs flattened input from conv layers
                "input_rank": 2,  # Expects 2D input (batch, features)
                "output_rank": 2,
            },
            LayerType.CONV2D: {
                "input_rank": 4,  # Expects 4D input (batch, height, width, channels)
                "output_rank": 4,
                "requires_spatial": True,
            },
            LayerType.CONV1D: {
                "input_rank": 3,  # Expects 3D input (batch, length, channels)
                "output_rank": 3,
                "requires_temporal": True,
            },
            LayerType.FLATTEN: {
                "input_rank": [3, 4],  # Can flatten 3D or 4D tensors
                "output_rank": 2,
            },
            LayerType.LSTM: {
                "input_rank": 3,  # Expects 3D input (batch, timesteps, features)
                "output_rank": [
                    2,
                    3,
                ],  # Can output 2D or 3D depending on return_sequences
                "requires_temporal": True,
            },
            LayerType.GRU: {
                "input_rank": 3,
                "output_rank": [2, 3],
                "requires_temporal": True,
            },
        }

    def _validate_kernel_size(self, value: Any) -> bool:
        """Validate kernel size parameter."""
        if isinstance(value, int):
            return 1 <= value <= 11
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            return all(isinstance(v, int) and 1 <= v <= 11 for v in value)
        return False

    def _validate_strides(self, value: Any) -> bool:
        """Validate strides parameter."""
        if isinstance(value, int):
            return 1 <= value <= 5
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            return all(isinstance(v, int) and 1 <= v <= 5 for v in value)
        return False

    def _validate_pool_size(self, value: Any) -> bool:
        """Validate pool size parameter."""
        if isinstance(value, int):
            return 1 <= value <= 8
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            return all(isinstance(v, int) and 1 <= v <= 8 for v in value)
        return False

    def _check_pruning_before_quantization(self, graph) -> bool:
        """Check if pruning comes before quantization in pipeline."""
        # This would check pipeline metadata for transformation order
        pipeline_order = graph.metadata.get("pipeline_transformations", [])
        pruning_idx = -1
        quantization_idx = -1

        for i, transform in enumerate(pipeline_order):
            if transform.get("type") == "pruning":
                pruning_idx = i
            elif transform.get("type") == "quantization":
                quantization_idx = i

        return (
            pruning_idx != -1
            and quantization_idx != -1
            and pruning_idx < quantization_idx
        )

    def _check_high_dropout_with_batchnorm(self, graph) -> bool:
        """Check for high dropout rates with batch normalization."""
        dropout_nodes = graph.get_layers_by_type(LayerType.DROPOUT)
        batchnorm_nodes = graph.get_layers_by_type(LayerType.BATCH_NORM)

        if not dropout_nodes or not batchnorm_nodes:
            return False

        # Check if any dropout layer has rate > 0.5 and is near a batch norm layer
        for dropout_node in dropout_nodes:
            dropout_rate = dropout_node.get_param("rate", 0.0)
            if dropout_rate > 0.5:
                # Check if there's a batch norm layer within 2 hops
                for batchnorm_node in batchnorm_nodes:
                    if self._nodes_within_distance(
                        graph,
                        dropout_node.node_id,
                        batchnorm_node.node_id,
                        max_distance=2,
                    ):
                        return True
        return False

    def _check_consecutive_dropout(self, graph) -> bool:
        """Check for too many consecutive dropout layers."""
        execution_order = graph.get_execution_order()
        consecutive_dropout = 0
        max_consecutive = 3

        for node_id in execution_order:
            node = graph.get_node(node_id)
            if node and node.layer_type == LayerType.DROPOUT:
                consecutive_dropout += 1
                if consecutive_dropout >= max_consecutive:
                    return True
            else:
                consecutive_dropout = 0

        return False

    def _nodes_within_distance(
        self, graph, node1_id: str, node2_id: str, max_distance: int
    ) -> bool:
        """Check if two nodes are within a certain distance in the graph."""
        visited = set()
        queue = [(node1_id, 0)]

        while queue:
            current_id, distance = queue.pop(0)

            if current_id == node2_id:
                return True

            if distance >= max_distance or current_id in visited:
                continue

            visited.add(current_id)
            current_node = graph.get_node(current_id)

            if current_node:
                for neighbor_id in current_node.input_nodes + current_node.output_nodes:
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, distance + 1))

        return False

    def get_parameter_range(
        self, layer_type: LayerType, param_name: str
    ) -> Optional[ParameterRange]:
        """Get parameter range for a specific layer type and parameter."""
        layer_params = self.parameter_ranges.get(layer_type, {})
        return layer_params.get(param_name)

    def is_sequence_forbidden(
        self, prev_layer: LayerType, curr_layer: LayerType
    ) -> bool:
        """Check if a layer sequence is forbidden."""
        return (prev_layer, curr_layer) in self.forbidden_sequences

    def get_compatibility_rule(self, layer_type: LayerType) -> Dict[str, Any]:
        """Get compatibility rules for a layer type."""
        return self.layer_compatibility_rules.get(layer_type, {})

    def update_device_constraints(self, constraints: DeviceConstraints) -> None:
        """Update device constraints."""
        self.device_constraints = constraints

    def add_custom_parameter_range(
        self, layer_type: LayerType, param_name: str, param_range: ParameterRange
    ) -> None:
        """Add or update a parameter range."""
        if layer_type not in self.parameter_ranges:
            self.parameter_ranges[layer_type] = {}
        self.parameter_ranges[layer_type][param_name] = param_range

    def add_forbidden_sequence(
        self, prev_layer: LayerType, curr_layer: LayerType
    ) -> None:
        """Add a forbidden layer sequence."""
        sequence = (prev_layer, curr_layer)
        if sequence not in self.forbidden_sequences:
            self.forbidden_sequences.append(sequence)


# Default configurations for different device types
def get_edge_device_config() -> ConstraintConfig:
    """Get configuration optimized for edge devices."""
    config = ConstraintConfig()
    config.device_constraints = DeviceConstraints(
        max_memory_mb=256.0,
        max_compute_ops=100000,
        max_tensor_size=100000,
        max_kernel_size=7,
        max_filters=256,
        max_units=1024,
        supported_dtypes=[DataType.FLOAT16, DataType.INT8, DataType.UINT8],
    )
    return config


def get_mobile_device_config() -> ConstraintConfig:
    """Get configuration optimized for mobile devices."""
    config = ConstraintConfig()
    config.device_constraints = DeviceConstraints(
        max_memory_mb=512.0,
        max_compute_ops=500000,
        max_tensor_size=500000,
        max_kernel_size=9,
        max_filters=512,
        max_units=2048,
        supported_dtypes=[DataType.FLOAT32, DataType.FLOAT16, DataType.INT8],
    )
    return config


def get_server_device_config() -> ConstraintConfig:
    """Get configuration for server/cloud deployment."""
    config = ConstraintConfig()
    config.device_constraints = DeviceConstraints(
        max_memory_mb=8192.0,
        max_compute_ops=10000000,
        max_tensor_size=10000000,
        max_kernel_size=15,
        max_filters=2048,
        max_units=8192,
        supported_dtypes=list(DataType),
    )
    return config
