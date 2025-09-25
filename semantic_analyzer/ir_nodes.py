"""
Intermediate Representation (IR) nodes and graph structures for semantic analysis.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .error_types import SourceLocation


class DataType(Enum):
    """Supported data types in the DSL."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"


class ActivationType(Enum):
    """Supported activation functions."""

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LEAKY_RELU = "leaky_relu"
    SWISH = "swish"
    GELU = "gelu"
    LINEAR = "linear"


@dataclass
class TensorShape:
    """Represents tensor shape with batch dimension handling."""

    dimensions: Tuple[int, ...]
    batch_size: Optional[int] = None

    def __post_init__(self):
        # Convert list to tuple if needed
        if isinstance(self.dimensions, list):
            self.dimensions = tuple(self.dimensions)

    @property
    def rank(self) -> int:
        """Number of dimensions (excluding batch)."""
        return len(self.dimensions)

    @property
    def total_elements(self) -> int:
        """Total number of elements (excluding batch)."""
        if not self.dimensions:
            return 0
        result = 1
        for dim in self.dimensions:
            if dim > 0:  # Handle dynamic dimensions
                result *= dim
        return result

    def with_batch(self, batch_size: int) -> "TensorShape":
        """Return new shape with batch dimension."""
        return TensorShape(
            dimensions=(batch_size,) + self.dimensions, batch_size=batch_size
        )

    def without_batch(self) -> "TensorShape":
        """Return shape without batch dimension."""
        if (
            self.batch_size is not None
            and self.dimensions
            and self.dimensions[0] == self.batch_size
        ):
            return TensorShape(dimensions=self.dimensions[1:])
        return TensorShape(dimensions=self.dimensions)

    def is_compatible_with(self, other: "TensorShape") -> bool:
        """Check if this shape is compatible with another for operations."""
        # Handle dynamic dimensions (-1)
        if len(self.dimensions) != len(other.dimensions):
            return False

        for dim1, dim2 in zip(self.dimensions, other.dimensions):
            if dim1 != -1 and dim2 != -1 and dim1 != dim2:
                return False
        return True

    def __str__(self) -> str:
        if self.batch_size is not None:
            return f"({self.batch_size}, {', '.join(map(str, self.dimensions))})"
        return f"({', '.join(map(str, self.dimensions))})"

    def __eq__(self, other) -> bool:
        if isinstance(other, tuple):
            return self.dimensions == other
        return isinstance(other, TensorShape) and self.dimensions == other.dimensions


@dataclass
class TensorInfo:
    """Complete tensor information including shape and data type."""

    shape: TensorShape
    dtype: DataType
    name: Optional[str] = None

    def memory_usage_bytes(self) -> int:
        """Calculate memory usage in bytes."""
        element_count = self.shape.total_elements
        if self.shape.batch_size:
            element_count *= self.shape.batch_size

        bytes_per_element = {
            DataType.FLOAT32: 4,
            DataType.FLOAT16: 2,
            DataType.INT32: 4,
            DataType.INT16: 2,
            DataType.INT8: 1,
            DataType.UINT8: 1,
            DataType.BOOL: 1,
        }

        return element_count * bytes_per_element.get(self.dtype, 4)


class LayerType(Enum):
    """Types of layers supported in the DSL."""

    INPUT = "Input"
    DENSE = "Dense"
    CONV2D = "Conv2D"
    CONV1D = "Conv1D"
    MAXPOOL2D = "MaxPool2D"
    AVGPOOL2D = "AvgPool2D"
    FLATTEN = "Flatten"
    DROPOUT = "Dropout"
    BATCH_NORM = "BatchNorm"
    LAYER_NORM = "LayerNorm"
    ACTIVATION = "Activation"
    CONCATENATE = "Concatenate"
    ADD = "Add"
    MULTIPLY = "Multiply"
    OUTPUT = "Output"
    LSTM = "LSTM"
    GRU = "GRU"
    EMBEDDING = "Embedding"
    ATTENTION = "Attention"


@dataclass
class IRNode:
    """Base class for IR nodes representing layers/operations."""

    node_id: str
    layer_type: LayerType
    name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_tensors: List[TensorInfo] = field(default_factory=list)
    output_tensors: List[TensorInfo] = field(default_factory=list)
    input_nodes: List[str] = field(default_factory=list)  # Node IDs
    output_nodes: List[str] = field(default_factory=list)  # Node IDs
    location: Optional[SourceLocation] = None
    device_constraints: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Get display name for the node."""
        return self.name or f"{self.layer_type.value}_{self.node_id}"

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get parameter value with default."""
        return self.parameters.get(name, default)

    def set_param(self, name: str, value: Any) -> None:
        """Set parameter value."""
        self.parameters[name] = value

    def add_input_tensor(self, tensor: TensorInfo) -> None:
        """Add input tensor."""
        self.input_tensors.append(tensor)

    def add_output_tensor(self, tensor: TensorInfo) -> None:
        """Add output tensor."""
        self.output_tensors.append(tensor)

    def connect_to(self, other_node: "IRNode") -> None:
        """Connect this node's output to another node's input."""
        if other_node.node_id not in self.output_nodes:
            self.output_nodes.append(other_node.node_id)
        if self.node_id not in other_node.input_nodes:
            other_node.input_nodes.append(self.node_id)


class IRGraph:
    """Represents the complete IR graph of the model."""

    def __init__(self):
        self.nodes: Dict[str, IRNode] = {}
        self.input_nodes: List[str] = []
        self.output_nodes: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def add_node(self, node: IRNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node

        # Track input/output nodes
        if node.layer_type == LayerType.INPUT:
            self.input_nodes.append(node.node_id)
        elif node.layer_type == LayerType.OUTPUT:
            self.output_nodes.append(node.node_id)

    def get_node(self, node_id: str) -> Optional[IRNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove node and update connections."""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Remove connections
        for input_id in node.input_nodes:
            input_node = self.nodes.get(input_id)
            if input_node and node_id in input_node.output_nodes:
                input_node.output_nodes.remove(node_id)

        for output_id in node.output_nodes:
            output_node = self.nodes.get(output_id)
            if output_node and node_id in output_node.input_nodes:
                output_node.input_nodes.remove(node_id)

        # Remove from special lists
        if node_id in self.input_nodes:
            self.input_nodes.remove(node_id)
        if node_id in self.output_nodes:
            self.output_nodes.remove(node_id)

        del self.nodes[node_id]

    def get_predecessors(self, node_id: str) -> List[IRNode]:
        """Get all predecessor nodes."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [
            self.nodes[pred_id] for pred_id in node.input_nodes if pred_id in self.nodes
        ]

    def get_successors(self, node_id: str) -> List[IRNode]:
        """Get all successor nodes."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [
            self.nodes[succ_id]
            for succ_id in node.output_nodes
            if succ_id in self.nodes
        ]

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order."""
        visited = set()
        temp_visited = set()
        result = []

        def visit(node_id: str) -> bool:
            if node_id in temp_visited:
                return False  # Cycle detected
            if node_id in visited:
                return True

            temp_visited.add(node_id)

            node = self.nodes.get(node_id)
            if node:
                for pred_id in node.input_nodes:
                    if not visit(pred_id):
                        return False

            temp_visited.remove(node_id)
            visited.add(node_id)
            result.append(node_id)
            return True

        # Visit all nodes
        for node_id in self.nodes:
            if node_id not in visited:
                if not visit(node_id):
                    return []  # Cycle detected

        return result

    def has_cycles(self) -> bool:
        """Check if the graph has cycles."""
        return len(self.topological_sort()) == 0

    def is_connected(self) -> bool:
        """Check if the graph is connected (ignoring direction)."""
        if not self.nodes:
            return True

        visited = set()
        start_node = next(iter(self.nodes.keys()))

        def dfs(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)

            node = self.nodes.get(node_id)
            if node:
                for neighbor_id in node.input_nodes + node.output_nodes:
                    if neighbor_id in self.nodes:
                        dfs(neighbor_id)

        dfs(start_node)
        return len(visited) == len(self.nodes)

    def get_execution_order(self) -> List[str]:
        """Get nodes in execution order (topological sort)."""
        return self.topological_sort()

    def calculate_total_memory_usage(self) -> int:
        """Calculate total memory usage of all tensors in bytes."""
        total_memory = 0
        for node in self.nodes.values():
            for tensor in node.input_tensors + node.output_tensors:
                total_memory += tensor.memory_usage_bytes()
        return total_memory

    def get_layers_by_type(self, layer_type: LayerType) -> List[IRNode]:
        """Get all nodes of a specific layer type."""
        return [node for node in self.nodes.values() if node.layer_type == layer_type]

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes.values())


# Factory functions for common layer types
def create_input_node(
    node_id: str,
    shape: TensorShape,
    dtype: DataType = DataType.FLOAT32,
    name: str = None,
    location: SourceLocation = None,
) -> IRNode:
    """Create an input node."""
    tensor = TensorInfo(
        shape=shape, dtype=dtype, name=f"{name}_input" if name else None
    )
    return IRNode(
        node_id=node_id,
        layer_type=LayerType.INPUT,
        name=name,
        output_tensors=[tensor],
        location=location,
    )


def create_dense_node(
    node_id: str,
    units: int,
    activation: ActivationType = ActivationType.LINEAR,
    use_bias: bool = True,
    name: str = None,
    location: SourceLocation = None,
) -> IRNode:
    """Create a dense/fully connected layer node."""
    return IRNode(
        node_id=node_id,
        layer_type=LayerType.DENSE,
        name=name,
        parameters={"units": units, "activation": activation, "use_bias": use_bias},
        location=location,
    )


def create_conv2d_node(
    node_id: str,
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = 1,
    padding: str = "valid",
    activation: ActivationType = ActivationType.LINEAR,
    use_bias: bool = True,
    name: str = None,
    location: SourceLocation = None,
) -> IRNode:
    """Create a 2D convolution layer node."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)

    return IRNode(
        node_id=node_id,
        layer_type=LayerType.CONV2D,
        name=name,
        parameters={
            "filters": filters,
            "kernel_size": kernel_size,
            "strides": strides,
            "padding": padding,
            "activation": activation,
            "use_bias": use_bias,
        },
        location=location,
    )
