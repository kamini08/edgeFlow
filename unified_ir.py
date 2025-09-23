"""Unified Intermediate Representation (UIR) for Multi-Framework Support

This module implements a framework-agnostic intermediate representation that can
represent ML models from different frameworks (TensorFlow, ONNX, PyTorch) in a
unified format. The UIR serves as a bridge between framework-specific models
and EdgeFlow's optimization and code generation pipeline.

Key Features:
- Framework-agnostic representation of computational graphs
- Support for tensor shapes, data types, and operations
- Metadata preservation for framework-specific information
- Extensible schema for new frameworks and operations
- Integration with EdgeFlow's existing IR system
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class FrameworkType(Enum):
    """Supported ML frameworks."""

    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TFLITE = "tflite"
    UNKNOWN = "unknown"


class DataType(Enum):
    """Supported data types across frameworks."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    BOOL = "bool"
    STRING = "string"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"


class OperationType(Enum):
    """Common ML operations across frameworks."""

    # Convolution operations
    CONV2D = "conv2d"
    CONV1D = "conv1d"
    CONV3D = "conv3d"
    DEPTHWISE_CONV2D = "depthwise_conv2d"
    SEPARABLE_CONV2D = "separable_conv2d"

    # Pooling operations
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"
    GLOBAL_MAX_POOL = "global_max_pool"
    GLOBAL_AVG_POOL = "global_avg_pool"

    # Activation functions
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    GELU = "gelu"
    SWISH = "swish"
    LEAKY_RELU = "leaky_relu"

    # Normalization
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    GROUP_NORM = "group_norm"
    INSTANCE_NORM = "instance_norm"

    # Dense/Linear operations
    DENSE = "dense"
    MATMUL = "matmul"

    # Reshape operations
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    FLATTEN = "flatten"
    SQUEEZE = "squeeze"
    UNSQUEEZE = "unsqueeze"

    # Element-wise operations
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    SQRT = "sqrt"
    ABS = "abs"

    # Reduction operations
    REDUCE_SUM = "reduce_sum"
    REDUCE_MEAN = "reduce_mean"
    REDUCE_MAX = "reduce_max"
    REDUCE_MIN = "reduce_min"

    # Concatenation and splitting
    CONCAT = "concat"
    SPLIT = "split"
    STACK = "stack"
    UNSTACK = "unstack"

    # Recurrent operations
    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"

    # Attention operations
    ATTENTION = "attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"

    # Custom/Unknown operations
    CUSTOM = "custom"
    UNKNOWN = "unknown"


@dataclass
class TensorShape:
    """Represents a tensor shape with support for dynamic dimensions."""

    dimensions: List[Union[int, str]]  # -1 or "?" for dynamic dimensions
    rank: int = field(init=False)

    def __post_init__(self):
        self.rank = len(self.dimensions)

    def is_dynamic(self) -> bool:
        """Check if the shape contains dynamic dimensions."""
        return any(
            dim == -1 or dim == "?" or isinstance(dim, str) for dim in self.dimensions
        )

    def to_static(self, dynamic_values: Optional[Dict[str, int]] = None) -> TensorShape:
        """Convert dynamic dimensions to static values."""
        if not self.is_dynamic():
            return self

        static_dims = []
        for dim in self.dimensions:
            if dim == -1 or dim == "?":
                static_dims.append(1)  # Default fallback
            elif isinstance(dim, str) and dynamic_values and dim in dynamic_values:
                static_dims.append(dynamic_values[dim])
            else:
                static_dims.append(dim)

        return TensorShape(static_dims)

    def __str__(self) -> str:
        return f"[{', '.join(str(d) for d in self.dimensions)}]"


@dataclass
class TensorInfo:
    """Information about a tensor in the computational graph."""

    name: str
    shape: TensorShape
    dtype: DataType
    framework_metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: {self.shape} {self.dtype.value}"


@dataclass
class OperationAttribute:
    """An attribute of an operation (parameter, configuration, etc.)."""

    name: str
    value: Any
    dtype: Optional[DataType] = None
    framework_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UIRNode:
    """A node in the unified intermediate representation graph."""

    node_id: str
    name: str
    operation_type: OperationType
    framework_type: FrameworkType
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict[str, OperationAttribute] = field(default_factory=dict)
    framework_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_attribute(
        self, name: str, value: Any, dtype: Optional[DataType] = None
    ) -> None:
        """Add an attribute to this node."""
        self.attributes[name] = OperationAttribute(name=name, value=value, dtype=dtype)

    def get_attribute(self, name: str, default: Any = None) -> Any:
        """Get an attribute value."""
        return self.attributes.get(name, OperationAttribute(name, default)).value


@dataclass
class UIRGraph:
    """Unified Intermediate Representation graph."""

    name: str
    framework_type: FrameworkType
    nodes: Dict[str, UIRNode] = field(default_factory=dict)
    tensors: Dict[str, TensorInfo] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(
        default_factory=list
    )  # (from_node, to_node, tensor_name)
    framework_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: UIRNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        logger.debug(f"Added UIR node: {node.node_id} ({node.operation_type.value})")

    def add_tensor(self, tensor: TensorInfo) -> None:
        """Add tensor information to the graph."""
        self.tensors[tensor.name] = tensor
        logger.debug(f"Added tensor: {tensor}")

    def add_edge(self, from_node_id: str, to_node_id: str, tensor_name: str) -> None:
        """Add an edge between two nodes."""
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError(f"Nodes {from_node_id} or {to_node_id} not found in graph")

        self.edges.append((from_node_id, to_node_id, tensor_name))
        logger.debug(f"Added edge: {from_node_id} -> {to_node_id} via {tensor_name}")

    def get_node_inputs(self, node_id: str) -> List[str]:
        """Get input tensor names for a node."""
        return [edge[2] for edge in self.edges if edge[1] == node_id]

    def get_node_outputs(self, node_id: str) -> List[str]:
        """Get output tensor names for a node."""
        return [edge[2] for edge in self.edges if edge[0] == node_id]

    def topological_sort(self) -> List[str]:
        """Perform topological sort to determine execution order."""
        visited = set()
        temp_visited = set()
        result = []

        def visit(node_id: str):
            if node_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node_id}")
            if node_id in visited:
                return

            temp_visited.add(node_id)

            # Visit all input nodes first
            for from_node, to_node, _ in self.edges:
                if to_node == node_id:
                    visit(from_node)

            temp_visited.remove(node_id)
            visited.add(node_id)
            result.append(node_id)

        # Visit all nodes
        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)

        return result

    def validate_graph(self) -> Tuple[bool, List[str]]:
        """Validate the UIR graph for correctness."""
        errors = []

        # Check for isolated nodes
        connected_nodes = set()
        for from_node, to_node, _ in self.edges:
            connected_nodes.add(from_node)
            connected_nodes.add(to_node)

        for node_id in self.nodes:
            if node_id not in connected_nodes:
                errors.append(f"Isolated node: {node_id}")

        # Check for cycles
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))

        # Check for missing tensor references
        for from_node, to_node, tensor_name in self.edges:
            if tensor_name not in self.tensors:
                errors.append(f"Edge references non-existent tensor: {tensor_name}")

        return len(errors) == 0, errors

    def get_graph_info(self) -> Dict[str, Any]:
        """Get comprehensive graph information."""
        return {
            "name": self.name,
            "framework_type": self.framework_type.value,
            "num_nodes": len(self.nodes),
            "num_tensors": len(self.tensors),
            "num_edges": len(self.edges),
            "operations": {
                op_type.value: [
                    node_id
                    for node_id, node in self.nodes.items()
                    if node.operation_type == op_type
                ]
                for op_type in OperationType
            },
            "tensor_shapes": {
                name: str(tensor.shape) for name, tensor in self.tensors.items()
            },
            "framework_metadata": self.framework_metadata,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the UIR graph to a dictionary representation."""
        return {
            "name": self.name,
            "framework_type": self.framework_type.value,
            "nodes": [
                {
                    "id": node.node_id,
                    "name": node.name,
                    "operation_type": node.operation_type.value,
                    "framework_type": node.framework_type.value,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "attributes": {
                        name: {
                            "value": attr.value,
                            "dtype": attr.dtype.value if attr.dtype else None,
                        }
                        for name, attr in node.attributes.items()
                    },
                    "framework_metadata": node.framework_metadata,
                }
                for node in self.nodes.values()
            ],
            "tensors": [
                {
                    "name": tensor.name,
                    "shape": tensor.shape.dimensions,
                    "dtype": tensor.dtype.value,
                    "framework_metadata": tensor.framework_metadata,
                }
                for tensor in self.tensors.values()
            ],
            "edges": [
                {"from": edge[0], "to": edge[1], "tensor": edge[2]}
                for edge in self.edges
            ],
            "framework_metadata": self.framework_metadata,
        }


class FrameworkParser(ABC):
    """Abstract base class for framework-specific parsers."""

    @abstractmethod
    def parse_model(self, model_path: str) -> UIRGraph:
        """Parse a model from the specific framework into UIR."""
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats for this framework."""
        pass

    @abstractmethod
    def get_framework_type(self) -> FrameworkType:
        """Get the framework type this parser handles."""
        pass


class UIRTransformation(ABC):
    """Abstract base class for UIR transformations."""

    @abstractmethod
    def transform(self, graph: UIRGraph) -> UIRGraph:
        """Apply transformation to the UIR graph."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this transformation."""
        pass


class UIRValidator(ABC):
    """Abstract base class for UIR validators."""

    @abstractmethod
    def validate(self, graph: UIRGraph) -> Tuple[bool, List[str]]:
        """Validate the UIR graph."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this validator."""
        pass


def create_uir_from_edgeflow_config(config: Dict[str, Any]) -> UIRGraph:
    """Create a UIR graph from EdgeFlow configuration.

    This function bridges the existing EdgeFlow configuration system
    with the new unified IR system.
    """
    model_path = config.get("model", "model.tflite")
    framework_type = FrameworkType.TFLITE  # Default for EdgeFlow

    # Determine framework type from model path
    if model_path.endswith((".h5", ".keras")):
        framework_type = FrameworkType.TENSORFLOW
    elif model_path.endswith(".onnx"):
        framework_type = FrameworkType.ONNX
    elif model_path.endswith(".pth"):
        framework_type = FrameworkType.PYTORCH
    elif model_path.endswith(".tflite"):
        framework_type = FrameworkType.TFLITE

    graph = UIRGraph(
        name="edgeflow_model",
        framework_type=framework_type,
        framework_metadata={
            "edgeflow_config": config,
            "model_path": model_path,
        },
    )

    # Create a simple model node representing the EdgeFlow model
    model_node = UIRNode(
        node_id="model_0",
        name="Main Model",
        operation_type=OperationType.CUSTOM,
        framework_type=framework_type,
        framework_metadata={
            "edgeflow_optimizations": {
                "quantize": config.get("quantize", "none"),
                "enable_fusion": config.get("enable_fusion", False),
                "enable_pruning": config.get("enable_pruning", False),
                "target_device": config.get("target_device", "cpu"),
            }
        },
    )

    # Add input and output tensors
    input_shape = config.get("input_shape", "1,224,224,3")
    shape_dims = [int(x.strip()) for x in input_shape.split(",")]

    input_tensor = TensorInfo(
        name="input",
        shape=TensorShape(shape_dims),
        dtype=DataType.FLOAT32,
        framework_metadata={"edgeflow_input": True},
    )

    output_tensor = TensorInfo(
        name="output",
        shape=TensorShape([shape_dims[0], 1000]),  # Assume 1000 classes
        dtype=DataType.FLOAT32,
        framework_metadata={"edgeflow_output": True},
    )

    graph.add_tensor(input_tensor)
    graph.add_tensor(output_tensor)
    graph.add_node(model_node)
    graph.add_edge("input", "model_0", "input")
    graph.add_edge("model_0", "output", "output")

    return graph


def merge_uir_graphs(graphs: List[UIRGraph]) -> UIRGraph:
    """Merge multiple UIR graphs into a single graph.

    This is useful for combining models from different frameworks
    or creating complex pipelines.
    """
    if not graphs:
        raise ValueError("Cannot merge empty list of graphs")

    # Use the first graph as the base
    merged = UIRGraph(
        name="merged_model",
        framework_type=FrameworkType.UNKNOWN,
        framework_metadata={"merged_from": [g.name for g in graphs]},
    )

    node_id_offset = 0
    tensor_name_offset = 0

    for graph in graphs:
        # Add nodes with offset IDs
        for node in graph.nodes.values():
            new_node = UIRNode(
                node_id=f"{node.node_id}_{node_id_offset}",
                name=node.name,
                operation_type=node.operation_type,
                framework_type=node.framework_type,
                inputs=node.inputs,
                outputs=node.outputs,
                attributes=node.attributes,
                framework_metadata={
                    **node.framework_metadata,
                    "original_graph": graph.name,
                    "original_node_id": node.node_id,
                },
            )
            merged.add_node(new_node)

        # Add tensors with offset names
        for tensor in graph.tensors.values():
            new_tensor = TensorInfo(
                name=f"{tensor.name}_{tensor_name_offset}",
                shape=tensor.shape,
                dtype=tensor.dtype,
                framework_metadata={
                    **tensor.framework_metadata,
                    "original_graph": graph.name,
                    "original_tensor_name": tensor.name,
                },
            )
            merged.add_tensor(new_tensor)

        # Add edges with updated references
        for from_node, to_node, tensor_name in graph.edges:
            merged.add_edge(
                f"{from_node}_{node_id_offset}",
                f"{to_node}_{node_id_offset}",
                f"{tensor_name}_{tensor_name_offset}",
            )

        node_id_offset += 1000  # Large offset to avoid conflicts
        tensor_name_offset += 1000

    return merged


if __name__ == "__main__":
    # Test the UIR system
    test_config = {
        "model": "test_model.tflite",
        "quantize": "int8",
        "target_device": "raspberry_pi",
        "input_shape": "1,224,224,3",
    }

    graph = create_uir_from_edgeflow_config(test_config)
    print("UIR Graph created:")
    print(f"  Name: {graph.name}")
    print(f"  Framework: {graph.framework_type.value}")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Tensors: {len(graph.tensors)}")
    print(f"  Edges: {len(graph.edges)}")

    # Validate the graph
    is_valid, errors = graph.validate_graph()
    print(f"  Valid: {is_valid}")
    if errors:
        print(f"  Errors: {errors}")

    # Get graph info
    info = graph.get_graph_info()
    print(f"  Graph info: {info}")
