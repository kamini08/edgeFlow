"""EdgeFlow Intermediate Representation (IR) System

This module implements a graph-based Intermediate Representation for EdgeFlow pipelines.
The IR represents the computation graph with nodes for models, preprocessing operations,
and I/O actions, enabling advanced optimizations and scheduling.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the IR graph."""

    INPUT = "input"
    MODEL = "model"
    PREPROCESS = "preprocess"
    POSTPROCESS = "postprocess"
    OUTPUT = "output"
    QUANTIZE = "quantize"
    FUSION = "fusion"
    SCHEDULE = "schedule"


@dataclass
class IRNode:
    """Base class for IR nodes.

    Extended with a flexible, framework-agnostic schema to support multi-framework
    conversion, validation, optimization, and provenance tracking. All new fields
    are optional and defaulted to maintain backward compatibility with existing
    usages that rely on ``properties``, ``dependencies`` and ``dependents``.
    """

    node_id: str
    name: str
    node_type: Optional[NodeType] = None

    # Canonical operator information (optional)
    op_type: Optional[str] = None  # e.g., "Conv2D", "Dense", "ReLU"
    input_shapes: Optional[List[List[int]]] = None  # list of shapes
    output_shapes: Optional[List[List[int]]] = None  # list of shapes
    params: Dict[str, Any] = field(default_factory=dict)  # operator params
    inputs: List[str] = field(default_factory=list)  # upstream node IDs or tensor names
    outputs: List[str] = field(
        default_factory=list
    )  # downstream node IDs or tensor names
    dtype: Optional[str] = None  # e.g., "float32", "int8"
    framework_origin: Optional[str] = None  # e.g., "onnx", "tf", "pytorch"
    device_constraints: Dict[str, Any] = field(default_factory=dict)  # hw limits

    # Provenance and transformation history
    provenance: List[Dict[str, Any]] = field(default_factory=list)

    # Backwards-compatible catch-all metadata containers
    properties: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Set default node type if not provided."""
        if self.node_type is None:
            self.node_type = NodeType.MODEL

    def log_transformation(self, description: str, **metadata: Any) -> None:
        """Append a provenance entry for auditing/debugging.

        Args:
            description: Human-friendly description of the change.
            **metadata: Optional structured details (tool version, timestamps, etc.).
        """
        entry = {"description": description}
        entry.update(metadata)
        self.provenance.append(entry)


@dataclass
class InputNode(IRNode):
    """Input node for data sources."""

    def __post_init__(self):
        self.node_type = NodeType.INPUT


@dataclass
class ModelNode(IRNode):
    """Model node for ML models."""

    def __post_init__(self):
        self.node_type = NodeType.MODEL


@dataclass
class PreprocessNode(IRNode):
    """Preprocessing operation node."""

    def __post_init__(self):
        self.node_type = NodeType.PREPROCESS


@dataclass
class PostprocessNode(IRNode):
    """Postprocessing operation node."""

    def __post_init__(self):
        self.node_type = NodeType.POSTPROCESS


@dataclass
class OutputNode(IRNode):
    """Output node for results."""

    def __post_init__(self):
        self.node_type = NodeType.OUTPUT


@dataclass
class QuantizeNode(IRNode):
    """Quantization operation node."""

    def __post_init__(self):
        self.node_type = NodeType.QUANTIZE


@dataclass
class FusionNode(IRNode):
    """Operation fusion node."""

    def __post_init__(self):
        self.node_type = NodeType.FUSION


@dataclass
class ScheduleNode(IRNode):
    """Scheduling optimization node."""

    def __post_init__(self):
        self.node_type = NodeType.SCHEDULE


class IRGraph:
    """Graph-based Intermediate Representation for EdgeFlow pipelines."""

    def __init__(self):
        self.nodes: Dict[str, IRNode] = {}
        self.edges: List[Tuple[str, str]] = []
        self.execution_order: List[str] = []
        self.optimization_passes: List[str] = []
        # Graph-level schema fields
        self.graph_inputs: List[str] = []  # node IDs where data enters the graph
        self.graph_outputs: List[str] = []  # node IDs where data exits the graph
        self.metadata: Dict[str, Any] = {}  # model/framework/device/global flags
        self.topology_info: Dict[str, Any] = {}  # optional connectivity statistics

    def add_node(self, node: IRNode) -> None:
        """Add a node to the IR graph."""
        self.nodes[node.node_id] = node
        node_type_str = node.node_type.value if node.node_type else "unknown"
        logger.debug(f"Added {node_type_str} node: {node.node_id}")

    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """Add a directed edge between two nodes."""
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError(f"Nodes {from_node_id} or {to_node_id} not found in graph")

        self.edges.append((from_node_id, to_node_id))

        # Update node dependencies
        self.nodes[to_node_id].dependencies.add(from_node_id)
        self.nodes[from_node_id].dependents.add(to_node_id)

        # Maintain canonical inputs/outputs lists for IR schema
        if from_node_id not in self.nodes[to_node_id].inputs:
            self.nodes[to_node_id].inputs.append(from_node_id)
        if to_node_id not in self.nodes[from_node_id].outputs:
            self.nodes[from_node_id].outputs.append(to_node_id)

        logger.debug(f"Added edge: {from_node_id} -> {to_node_id}")

    def get_dependencies(self, node_id: str) -> List[str]:
        """Get direct dependencies of a node."""
        return list(self.nodes[node_id].dependencies)

    def get_dependents(self, node_id: str) -> List[str]:
        """Get direct dependents of a node."""
        return list(self.nodes[node_id].dependents)

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

            # Visit all dependencies first
            for dep in self.get_dependencies(node_id):
                visit(dep)

            temp_visited.remove(node_id)
            visited.add(node_id)
            result.append(node_id)

        # Visit all nodes
        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)

        self.execution_order = result
        return result

    def validate_graph(self) -> Tuple[bool, List[str]]:
        """Validate the IR graph for correctness."""
        errors = []

        # Check for isolated nodes
        for node_id, node in self.nodes.items():
            if not node.dependencies and not node.dependents:
                errors.append(f"Isolated node: {node_id}")

        # Check for cycles
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))

        # Check for missing dependencies
        for from_id, to_id in self.edges:
            if from_id not in self.nodes or to_id not in self.nodes:
                errors.append(
                    f"Edge references non-existent node: {from_id} -> {to_id}"
                )

        return len(errors) == 0, errors

    def get_execution_plan(self) -> List[Dict[str, Any]]:
        """Get detailed execution plan for the graph."""
        if not self.execution_order:
            self.topological_sort()

        plan = []
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            plan.append(
                {
                    "node_id": node_id,
                    "node_type": node.node_type.value if node.node_type else "unknown",
                    "name": node.name,
                    "properties": node.properties,
                    "dependencies": list(node.dependencies),
                    "dependents": list(node.dependents),
                }
            )
        return plan

    def get_graph_info(self) -> Dict[str, Any]:
        """Get comprehensive graph information."""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "execution_order": self.execution_order,
            "optimization_passes": self.optimization_passes,
            "nodes_by_type": {
                node_type.value: [
                    node_id
                    for node_id, node in self.nodes.items()
                    if node.node_type == node_type
                ]
                for node_type in NodeType
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the IR graph to a dictionary representation."""
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "name": node.name,
                    "node_type": node.node_type.value if node.node_type else "unknown",
                    # Canonical operator schema
                    "op_type": node.op_type,
                    "input_shapes": node.input_shapes,
                    "output_shapes": node.output_shapes,
                    "params": dict(node.params),
                    "inputs": list(node.inputs),
                    "outputs": list(node.outputs),
                    "dtype": node.dtype,
                    "framework_origin": node.framework_origin,
                    "device_constraints": dict(node.device_constraints),
                    "provenance": list(node.provenance),
                    # Back-compat & graph relations
                    "properties": dict(getattr(node, "properties", {})),
                    "dependencies": list(getattr(node, "dependencies", [])),
                    "dependents": list(getattr(node, "dependents", [])),
                }
                for node in self.nodes.values()
            ],
            "edges": [{"from": edge[0], "to": edge[1]} for edge in self.edges],
            "execution_order": list(self.execution_order),
            "optimization_passes": list(self.optimization_passes),
            # Graph-level schema
            "graph_inputs": list(self.graph_inputs),
            "graph_outputs": list(self.graph_outputs),
            "metadata": dict(self.metadata),
            "topology_info": dict(self.topology_info),
        }


class IRTransformation(ABC):
    """Abstract base class for IR transformations."""

    @abstractmethod
    def transform(self, graph: IRGraph) -> IRGraph:
        """Apply transformation to the IR graph."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this transformation."""
        pass


class QuantizationPass(IRTransformation):
    """Transformation pass for quantization optimization."""

    def transform(self, graph: IRGraph) -> IRGraph:
        """Apply quantization optimizations to the graph."""
        logger.info("Applying quantization pass")

        # Find model nodes that need quantization
        model_nodes = [
            node for node in graph.nodes.values() if node.node_type == NodeType.MODEL
        ]

        for model_node in model_nodes:
            quantize_type = model_node.properties.get("quantize", "none")
            if quantize_type != "none":
                # Create quantization node
                quantize_id = f"quantize_{model_node.node_id}"
                quantize_node = QuantizeNode(
                    node_id=quantize_id,
                    name=f"Quantize {model_node.name}",
                    properties={"quantization_type": quantize_type},
                )

                graph.add_node(quantize_node)

                # Get all dependencies of the model node before making changes
                dependencies_to_update = list(model_node.dependencies)

                # Add edge from quantize to model
                graph.add_edge(quantize_id, model_node.node_id)

                # Update all dependencies to point to quantize instead of model
                for dep in dependencies_to_update:
                    # Only add edge if it's not the quantize node itself (avoid self-loops)
                    if dep != quantize_id:
                        graph.add_edge(dep, quantize_id)

                    # Remove the old edge from dependency to model
                    if (dep, model_node.node_id) in graph.edges:
                        graph.edges.remove((dep, model_node.node_id))

                    # Update node dependency relationships
                    model_node.dependencies.discard(dep)
                    if dep in graph.nodes:
                        graph.nodes[dep].dependents.discard(model_node.node_id)

        graph.optimization_passes.append("quantization")
        return graph

    def get_name(self) -> str:
        return "quantization_pass"


class FusionPass(IRTransformation):
    """Transformation pass for operation fusion."""

    def transform(self, graph: IRGraph) -> IRGraph:
        """Apply operation fusion optimizations."""
        logger.info("Applying fusion pass")

        # Find model nodes that can be fused
        model_nodes = [
            node for node in graph.nodes.values() if node.node_type == NodeType.MODEL
        ]

        for model_node in model_nodes:
            fusion_enabled = model_node.properties.get("enable_fusion", False)
            fusion_id = f"fusion_{model_node.node_id}"

            # Check if fusion node already exists and fusion is enabled
            if fusion_enabled and fusion_id not in graph.nodes:
                fusion_node = FusionNode(
                    node_id=fusion_id,
                    name=f"Fusion {model_node.name}",
                    properties={
                        "fusion_type": "conv_bn_relu",
                        "target_ops": ["conv2d", "batch_norm", "relu"],
                    },
                )

                graph.add_node(fusion_node)

                # Get all dependents of the model node before making changes
                dependents_to_update = list(model_node.dependents)

                # Add edge from model to fusion
                graph.add_edge(model_node.node_id, fusion_id)

                # Update all dependents to point to fusion instead of model
                for dep in dependents_to_update:
                    # Add edge from fusion to dependent
                    graph.add_edge(fusion_id, dep)

                    # Remove the old edge from model to dependent
                    if (model_node.node_id, dep) in graph.edges:
                        graph.edges.remove((model_node.node_id, dep))

                    # Update node dependency relationships
                    model_node.dependents.discard(dep)
                    if dep in graph.nodes:
                        graph.nodes[dep].dependencies.discard(model_node.node_id)

        graph.optimization_passes.append("fusion")
        return graph

    def get_name(self) -> str:
        return "fusion_pass"


class SchedulingPass(IRTransformation):
    """Transformation pass for device-specific scheduling."""

    def transform(self, graph: IRGraph) -> IRGraph:
        """Apply scheduling optimizations for target device."""
        logger.info("Applying scheduling pass")

        # Find all model nodes and add scheduling information
        model_nodes = [
            node for node in graph.nodes.values() if node.node_type == NodeType.MODEL
        ]

        for model_node in model_nodes:
            target_device = model_node.properties.get("target_device", "cpu")

            if target_device != "cpu":
                schedule_id = f"schedule_{model_node.node_id}"
                schedule_node = ScheduleNode(
                    node_id=schedule_id,
                    name=f"Schedule {model_node.name}",
                    properties={
                        "target_device": target_device,
                        "memory_optimization": True,
                        "parallel_execution": target_device in ["gpu", "tpu"],
                    },
                )

                graph.add_node(schedule_node)

                # Get all dependents of the model node before making changes
                dependents_to_update = list(model_node.dependents)

                # Add edge from model to schedule
                graph.add_edge(model_node.node_id, schedule_id)

                # Update all dependents to point to schedule instead of model
                for dep in dependents_to_update:
                    # Add edge from schedule to dependent
                    graph.add_edge(schedule_id, dep)

                    # Remove the old edge from model to dependent
                    if (model_node.node_id, dep) in graph.edges:
                        graph.edges.remove((model_node.node_id, dep))

                    # Update node dependency relationships
                    model_node.dependents.discard(dep)
                    if dep in graph.nodes:
                        graph.nodes[dep].dependencies.discard(model_node.node_id)

        graph.optimization_passes.append("scheduling")
        return graph

    def get_name(self) -> str:
        return "scheduling_pass"


class IRBuilder:
    """Builder for creating IR graphs from EdgeFlow configurations."""

    def build_from_config(self, config: Dict[str, Any]) -> IRGraph:
        """Build IR graph from EdgeFlow configuration."""
        logger.info("Building IR graph from configuration")

        graph = IRGraph()
        # Populate graph-level metadata
        graph.metadata = {
            "source_framework": config.get("source_framework", None),
            "opset_version": config.get("opset_version", None),
            "target_device": config.get("target_device", "cpu"),
            "optimization_flags": {
                "quantize": config.get("quantize", "none"),
                "enable_fusion": config.get("enable_fusion", False),
            },
            "deployment_constraints": {
                "memory_limit": config.get("memory_limit", None),
                "buffer_size": config.get("buffer_size", None),
            },
        }

        # Create input node
        input_shape_value = config.get("input_shape", "1,224,224,3")
        input_node = InputNode(
            node_id="input_0",
            name="Input Data",
            properties={
                "input_shape": input_shape_value,
                "data_type": "float32",
            },
            op_type="Input",
            input_shapes=[],
            output_shapes=[
                [
                    int(dim) if str(dim).isdigit() else -1
                    for dim in str(input_shape_value).split(",")
                ]
            ],
            dtype="float32",
            framework_origin=graph.metadata.get("source_framework"),
        )
        graph.add_node(input_node)

        # Create model node
        model_node = ModelNode(
            node_id="model_0",
            name="Main Model",
            properties={
                "model_path": config.get("model", "model.tflite"),
                "quantize": config.get("quantize", "none"),
                "target_device": config.get("target_device", "cpu"),
                "enable_fusion": config.get("enable_fusion", False),
            },
            op_type="Model",
            input_shapes=[
                [
                    int(dim) if str(dim).isdigit() else -1
                    for dim in str(input_shape_value).split(",")
                ]
            ],
            output_shapes=None,
            dtype=None,
            framework_origin=graph.metadata.get("source_framework"),
        )
        graph.add_node(model_node)

        # Create output node
        output_node = OutputNode(
            node_id="output_0",
            name="Output Results",
            properties={
                "output_format": config.get("output_format", "tensor"),
            },
            op_type="Output",
            input_shapes=None,
            output_shapes=[],
            dtype=None,
            framework_origin=graph.metadata.get("source_framework"),
        )
        graph.add_node(output_node)

        # Add edges
        graph.add_edge("input_0", "model_0")
        graph.add_edge("model_0", "output_0")

        # Add preprocessing if specified
        if config.get("preprocess"):
            preprocess_node = PreprocessNode(
                node_id="preprocess_0",
                name="Preprocessing",
                properties=config.get("preprocess", {}),
            )
            graph.add_node(preprocess_node)
            graph.add_edge("input_0", "preprocess_0")
            graph.add_edge("preprocess_0", "model_0")
            # Remove direct input->model edge
            graph.edges.remove(("input_0", "model_0"))
            model_node.dependencies.discard("input_0")
            input_node.dependents.discard("model_0")

        # Add postprocessing if specified
        if config.get("postprocess"):
            postprocess_node = PostprocessNode(
                node_id="postprocess_0",
                name="Postprocessing",
                properties=config.get("postprocess", {}),
            )
            graph.add_node(postprocess_node)
            graph.add_edge("model_0", "postprocess_0")
            graph.add_edge("postprocess_0", "output_0")
            # Remove direct model->output edge
            graph.edges.remove(("model_0", "output_0"))
            output_node.dependencies.discard("model_0")
            model_node.dependents.discard("output_0")

        # Perform topological sort
        graph.topological_sort()

        # Set graph IO after edges exist
        graph.graph_inputs = ["input_0"]
        graph.graph_outputs = ["output_0"]

        # Simple topology info
        graph.topology_info = {
            "num_nodes": len(graph.nodes),
            "num_edges": len(graph.edges),
            "has_cycles": False,
        }

        logger.info(
            f"Built IR graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
        )
        return graph


def create_ir_from_config(config: Dict[str, Any]) -> IRGraph:
    """Create IR graph from configuration."""
    builder = IRBuilder()
    return builder.build_from_config(config)


def optimize_ir_graph(graph: IRGraph, config: Dict[str, Any]) -> IRGraph:
    """Apply optimizations to IR graph."""
    logger.info("Optimizing IR graph")

    # Apply quantization pass
    if config.get("quantize") in ("int8", "float16"):
        quant_pass = QuantizationPass()
        graph = quant_pass.transform(graph)

    # Apply fusion pass
    if config.get("enable_fusion", False):
        fusion_pass = FusionPass()
        graph = fusion_pass.transform(graph)

    # Apply scheduling pass
    if config.get("target_device", "cpu") != "cpu":
        schedule_pass = SchedulingPass()
        graph = schedule_pass.transform(graph)

    return graph
