"""Cross-framework optimization passes for UIR graphs.

This module implements optimization passes that work on the unified intermediate
representation (UIR) to apply cross-framework optimizations such as quantization,
pruning, operator fusion, and hardware-specific tuning. These passes are
framework-agnostic and can be applied to models from any supported framework.
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
    UIRTransformation,
)

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimizations that can be applied."""

    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    FUSION = "fusion"
    MEMORY_OPTIMIZATION = "memory_optimization"
    HARDWARE_SPECIFIC = "hardware_specific"
    GRAPH_OPTIMIZATION = "graph_optimization"


class QuantizationType(Enum):
    """Types of quantization."""

    INT8 = "int8"
    FLOAT16 = "float16"
    DYNAMIC = "dynamic"
    NONE = "none"


@dataclass
class OptimizationResult:
    """Result of an optimization pass."""

    optimization_type: OptimizationType
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class QuantizationPass(UIRTransformation):
    """Quantization optimization pass for UIR graphs."""

    def __init__(self, quantization_type: QuantizationType = QuantizationType.INT8):
        self.quantization_type = quantization_type
        self.name = f"quantization_pass_{quantization_type.value}"

    def transform(self, graph: UIRGraph) -> UIRGraph:
        """Apply quantization to the UIR graph."""
        logger.info(f"Applying {self.quantization_type.value} quantization pass")

        if self.quantization_type == QuantizationType.NONE:
            return graph

        # Create quantized graph
        quantized_graph = UIRGraph(
            name=f"{graph.name}_quantized_{self.quantization_type.value}",
            framework_type=graph.framework_type,
            framework_metadata={
                **graph.framework_metadata,
                "quantization_type": self.quantization_type.value,
                "quantization_applied": True,
            },
        )

        # Copy tensors with quantization
        for tensor_name, tensor in graph.tensors.items():
            quantized_tensor = self._quantize_tensor(tensor)
            quantized_graph.add_tensor(quantized_tensor)

        # Apply quantization to nodes
        for node_id, node in graph.nodes.items():
            quantized_node = self._quantize_node(node)
            quantized_graph.add_node(quantized_node)

        # Copy edges
        for edge in graph.edges:
            quantized_graph.add_edge(*edge)

        return quantized_graph

    def _quantize_tensor(self, tensor: TensorInfo) -> TensorInfo:
        """Quantize a tensor."""
        # Determine target data type based on quantization type
        if self.quantization_type == QuantizationType.INT8:
            target_dtype = DataType.INT8
        elif self.quantization_type == QuantizationType.FLOAT16:
            target_dtype = DataType.FLOAT16
        else:
            target_dtype = tensor.dtype

        quantized_tensor = TensorInfo(
            name=tensor.name,
            shape=tensor.shape,
            dtype=target_dtype,
            framework_metadata={
                **tensor.framework_metadata,
                "quantized": True,
                "original_dtype": tensor.dtype.value,
                "quantization_type": self.quantization_type.value,
            },
        )

        return quantized_tensor

    def _quantize_node(self, node: UIRNode) -> UIRNode:
        """Quantize a node."""
        quantized_node = UIRNode(
            node_id=f"{node.node_id}_quantized",
            name=f"{node.name}_quantized",
            operation_type=node.operation_type,
            framework_type=node.framework_type,
            inputs=node.inputs,
            outputs=node.outputs,
            attributes=node.attributes.copy(),
            framework_metadata={
                **node.framework_metadata,
                "quantized": True,
                "quantization_type": self.quantization_type.value,
            },
        )

        # Add quantization-specific attributes
        if self.quantization_type == QuantizationType.INT8:
            quantized_node.add_attribute("quantization_scale", 1.0)
            quantized_node.add_attribute("quantization_zero_point", 0)
            quantized_node.add_attribute("quantization_min", -128)
            quantized_node.add_attribute("quantization_max", 127)
        elif self.quantization_type == QuantizationType.FLOAT16:
            quantized_node.add_attribute("precision", "half")

        return quantized_node

    def get_name(self) -> str:
        return self.name


class PruningPass(UIRTransformation):
    """Pruning optimization pass for UIR graphs."""

    def __init__(self, sparsity: float = 0.5, structured: bool = True):
        self.sparsity = sparsity
        self.structured = structured
        self.name = (
            f"pruning_pass_{sparsity}_{'structured' if structured else 'unstructured'}"
        )

    def transform(self, graph: UIRGraph) -> UIRGraph:
        """Apply pruning to the UIR graph."""
        logger.info(
            f"Applying pruning pass (sparsity: {self.sparsity}, structured: {self.structured})"
        )

        # Create pruned graph
        pruned_graph = UIRGraph(
            name=f"{graph.name}_pruned_{self.sparsity}",
            framework_type=graph.framework_type,
            framework_metadata={
                **graph.framework_metadata,
                "pruning_sparsity": self.sparsity,
                "pruning_structured": self.structured,
                "pruning_applied": True,
            },
        )

        # Copy tensors
        for tensor_name, tensor in graph.tensors.items():
            pruned_graph.add_tensor(tensor)

        # Apply pruning to nodes
        for node_id, node in graph.nodes.items():
            pruned_node = self._prune_node(node)
            pruned_graph.add_node(pruned_node)

        # Copy edges
        for edge in graph.edges:
            pruned_graph.add_edge(*edge)

        return pruned_graph

    def _prune_node(self, node: UIRNode) -> UIRNode:
        """Apply pruning to a node."""
        # Only prune certain operation types
        prunable_ops = {
            OperationType.CONV2D,
            OperationType.DENSE,
            OperationType.DEPTHWISE_CONV2D,
        }

        if node.operation_type not in prunable_ops:
            return node

        pruned_node = UIRNode(
            node_id=f"{node.node_id}_pruned",
            name=f"{node.name}_pruned",
            operation_type=node.operation_type,
            framework_type=node.framework_type,
            inputs=node.inputs,
            outputs=node.outputs,
            attributes=node.attributes.copy(),
            framework_metadata={
                **node.framework_metadata,
                "pruned": True,
                "pruning_sparsity": self.sparsity,
                "pruning_structured": self.structured,
            },
        )

        # Add pruning-specific attributes
        pruned_node.add_attribute("sparsity", self.sparsity)
        pruned_node.add_attribute("structured_pruning", self.structured)
        pruned_node.add_attribute("pruning_mask", f"mask_{node.node_id}")

        return pruned_node

    def get_name(self) -> str:
        return self.name


class FusionPass(UIRTransformation):
    """Operator fusion optimization pass for UIR graphs."""

    def __init__(self):
        self.name = "fusion_pass"
        self.fusion_patterns = self._create_fusion_patterns()

    def _create_fusion_patterns(self) -> List[Tuple[List[OperationType], str]]:
        """Create fusion patterns for common operation sequences."""
        return [
            # Conv2D + BatchNorm + ReLU
            (
                [OperationType.CONV2D, OperationType.BATCH_NORM, OperationType.RELU],
                "conv_bn_relu",
            ),
            # Conv2D + ReLU
            ([OperationType.CONV2D, OperationType.RELU], "conv_relu"),
            # Dense + ReLU
            ([OperationType.DENSE, OperationType.RELU], "dense_relu"),
            # Dense + BatchNorm + ReLU
            (
                [OperationType.DENSE, OperationType.BATCH_NORM, OperationType.RELU],
                "dense_bn_relu",
            ),
            # Add + ReLU
            ([OperationType.ADD, OperationType.RELU], "add_relu"),
            # Mul + Add (element-wise)
            ([OperationType.MUL, OperationType.ADD], "mul_add"),
        ]

    def transform(self, graph: UIRGraph) -> UIRGraph:
        """Apply operator fusion to the UIR graph."""
        logger.info("Applying operator fusion pass")

        # Create fused graph
        fused_graph = UIRGraph(
            name=f"{graph.name}_fused",
            framework_type=graph.framework_type,
            framework_metadata={
                **graph.framework_metadata,
                "fusion_applied": True,
            },
        )

        # Copy tensors
        for tensor_name, tensor in graph.tensors.items():
            fused_graph.add_tensor(tensor)

        # Find and apply fusion opportunities
        fused_nodes = self._find_and_apply_fusions(graph)

        # Add fused nodes
        for node in fused_nodes:
            fused_graph.add_node(node)

        # Copy remaining edges
        for edge in graph.edges:
            fused_graph.add_edge(*edge)

        return fused_graph

    def _find_and_apply_fusions(self, graph: UIRGraph) -> List[UIRNode]:
        """Find and apply fusion opportunities."""
        fused_nodes = []
        processed_nodes = set()

        # Get execution order
        execution_order = graph.topological_sort()

        for node_id in execution_order:
            if node_id in processed_nodes:
                continue

            node = graph.nodes[node_id]

            # Try to find fusion patterns starting from this node
            fusion_result = self._try_fuse_node(node, graph, execution_order)

            if fusion_result:
                fused_node, consumed_nodes = fusion_result
                fused_nodes.append(fused_node)
                processed_nodes.update(consumed_nodes)
            else:
                fused_nodes.append(node)
                processed_nodes.add(node_id)

        return fused_nodes

    def _try_fuse_node(
        self, start_node: UIRNode, graph: UIRGraph, execution_order: List[str]
    ) -> Optional[Tuple[UIRNode, Set[str]]]:
        """Try to fuse a node with its successors."""
        for pattern_ops, fusion_name in self.fusion_patterns:
            if start_node.operation_type != pattern_ops[0]:
                continue

            # Try to match the pattern
            matched_nodes = [start_node]
            consumed_nodes = {start_node.node_id}
            current_node = start_node

            for i in range(1, len(pattern_ops)):
                next_node = self._find_next_node(
                    current_node, pattern_ops[i], graph, execution_order
                )
                if next_node is None:
                    break

                matched_nodes.append(next_node)
                consumed_nodes.add(next_node.node_id)
                current_node = next_node

            # If we matched the full pattern, create fused node
            if len(matched_nodes) == len(pattern_ops):
                fused_node = self._create_fused_node(matched_nodes, fusion_name)
                return fused_node, consumed_nodes

        return None

    def _find_next_node(
        self,
        current_node: UIRNode,
        target_op_type: OperationType,
        graph: UIRGraph,
        execution_order: List[str],
    ) -> Optional[UIRNode]:
        """Find the next node in the execution order that matches the target operation type."""
        current_index = execution_order.index(current_node.node_id)

        for i in range(current_index + 1, len(execution_order)):
            next_node_id = execution_order[i]
            next_node = graph.nodes[next_node_id]

            # Check if this node is connected to the current node
            if self._are_nodes_connected(current_node, next_node, graph):
                if next_node.operation_type == target_op_type:
                    return next_node

        return None

    def _are_nodes_connected(
        self, from_node: UIRNode, to_node: UIRNode, graph: UIRGraph
    ) -> bool:
        """Check if two nodes are directly connected."""
        for edge in graph.edges:
            if edge[0] == from_node.node_id and edge[1] == to_node.node_id:
                return True
        return False

    def _create_fused_node(self, nodes: List[UIRNode], fusion_name: str) -> UIRNode:
        """Create a fused node from a list of nodes."""
        # Use the first node as the base
        base_node = nodes[0]

        fused_node = UIRNode(
            node_id=f"fused_{fusion_name}_{base_node.node_id}",
            name=f"Fused_{fusion_name}_{base_node.name}",
            operation_type=OperationType.CUSTOM,  # Fused operations are custom
            framework_type=base_node.framework_type,
            inputs=base_node.inputs,
            outputs=nodes[-1].outputs,  # Outputs from the last node
            attributes=base_node.attributes.copy(),
            framework_metadata={
                **base_node.framework_metadata,
                "fused": True,
                "fusion_name": fusion_name,
                "fused_operations": [node.operation_type.value for node in nodes],
                "fused_node_ids": [node.node_id for node in nodes],
            },
        )

        # Add fusion-specific attributes
        fused_node.add_attribute("fusion_type", fusion_name)
        fused_node.add_attribute("num_fused_ops", len(nodes))
        fused_node.add_attribute(
            "fused_operation_types",
            [op.value for op in [node.operation_type for node in nodes]],
        )

        return fused_node

    def get_name(self) -> str:
        return self.name


class MemoryOptimizationPass(UIRTransformation):
    """Memory optimization pass for UIR graphs."""

    def __init__(self):
        self.name = "memory_optimization_pass"

    def transform(self, graph: UIRGraph) -> UIRGraph:
        """Apply memory optimizations to the UIR graph."""
        logger.info("Applying memory optimization pass")

        # Create memory-optimized graph
        mem_opt_graph = UIRGraph(
            name=f"{graph.name}_mem_opt",
            framework_type=graph.framework_type,
            framework_metadata={
                **graph.framework_metadata,
                "memory_optimized": True,
            },
        )

        # Copy tensors with memory optimizations
        for tensor_name, tensor in graph.tensors.items():
            mem_opt_tensor = self._optimize_tensor_memory(tensor)
            mem_opt_graph.add_tensor(mem_opt_tensor)

        # Apply memory optimizations to nodes
        for node_id, node in graph.nodes.items():
            mem_opt_node = self._optimize_node_memory(node)
            mem_opt_graph.add_node(mem_opt_node)

        # Copy edges
        for edge in graph.edges:
            mem_opt_graph.add_edge(*edge)

        return mem_opt_graph

    def _optimize_tensor_memory(self, tensor: TensorInfo) -> TensorInfo:
        """Optimize tensor memory usage."""
        # For now, just copy the tensor
        # In a full implementation, this would apply memory layout optimizations
        return tensor

    def _optimize_node_memory(self, node: UIRNode) -> UIRNode:
        """Optimize node memory usage."""
        mem_opt_node = UIRNode(
            node_id=f"{node.node_id}_mem_opt",
            name=f"{node.name}_mem_opt",
            operation_type=node.operation_type,
            framework_type=node.framework_type,
            inputs=node.inputs,
            outputs=node.outputs,
            attributes=node.attributes.copy(),
            framework_metadata={
                **node.framework_metadata,
                "memory_optimized": True,
            },
        )

        # Add memory optimization attributes
        mem_opt_node.add_attribute("memory_layout", "optimized")
        mem_opt_node.add_attribute("inplace_operations", True)
        mem_opt_node.add_attribute("memory_pooling", True)

        return mem_opt_node

    def get_name(self) -> str:
        return self.name


class HardwareSpecificOptimizationPass(UIRTransformation):
    """Hardware-specific optimization pass for UIR graphs."""

    def __init__(self, target_device: str):
        self.target_device = target_device
        self.name = f"hardware_specific_pass_{target_device}"
        self.device_capabilities = self._get_device_capabilities()

    def _get_device_capabilities(self) -> Dict[str, Any]:
        """Get capabilities for the target device."""
        capabilities = {
            "raspberry_pi": {
                "neon_simd": True,
                "arm_cortex_a72": True,
                "memory_bandwidth": "low",
                "cache_size": "small",
                "preferred_data_types": [DataType.INT8, DataType.FLOAT32],
                "vectorization_width": 4,
            },
            "jetson_nano": {
                "neon_simd": True,
                "cuda_cores": 128,
                "arm_cortex_a57": True,
                "memory_bandwidth": "medium",
                "cache_size": "medium",
                "preferred_data_types": [
                    DataType.FLOAT16,
                    DataType.INT8,
                    DataType.FLOAT32,
                ],
                "vectorization_width": 8,
            },
            "cortex_m4": {
                "dsp_instructions": True,
                "arm_cortex_m4": True,
                "memory_bandwidth": "very_low",
                "cache_size": "tiny",
                "preferred_data_types": [DataType.INT8, DataType.INT16],
                "vectorization_width": 2,
            },
            "gpu": {
                "cuda_cores": 1024,
                "memory_bandwidth": "high",
                "cache_size": "large",
                "preferred_data_types": [DataType.FLOAT16, DataType.FLOAT32],
                "vectorization_width": 32,
            },
        }
        return capabilities.get(self.target_device, capabilities["cpu"])

    def transform(self, graph: UIRGraph) -> UIRGraph:
        """Apply hardware-specific optimizations to the UIR graph."""
        logger.info(
            f"Applying hardware-specific optimization pass for {self.target_device}"
        )

        # Create hardware-optimized graph
        hw_opt_graph = UIRGraph(
            name=f"{graph.name}_hw_{self.target_device}",
            framework_type=graph.framework_type,
            framework_metadata={
                **graph.framework_metadata,
                "hardware_optimized": True,
                "target_device": self.target_device,
            },
        )

        # Copy tensors
        for tensor_name, tensor in graph.tensors.items():
            hw_opt_graph.add_tensor(tensor)

        # Apply hardware-specific optimizations to nodes
        for node_id, node in graph.nodes.items():
            hw_opt_node = self._optimize_for_hardware(node)
            hw_opt_graph.add_node(hw_opt_node)

        # Copy edges
        for edge in graph.edges:
            hw_opt_graph.add_edge(*edge)

        return hw_opt_graph

    def _optimize_for_hardware(self, node: UIRNode) -> UIRNode:
        """Optimize a node for the target hardware."""
        hw_opt_node = UIRNode(
            node_id=f"{node.node_id}_hw_{self.target_device}",
            name=f"{node.name}_hw_{self.target_device}",
            operation_type=node.operation_type,
            framework_type=node.framework_type,
            inputs=node.inputs,
            outputs=node.outputs,
            attributes=node.attributes.copy(),
            framework_metadata={
                **node.framework_metadata,
                "hardware_optimized": True,
                "target_device": self.target_device,
            },
        )

        # Add hardware-specific attributes
        caps = self.device_capabilities
        hw_opt_node.add_attribute("target_device", self.target_device)
        hw_opt_node.add_attribute(
            "vectorization_width", caps.get("vectorization_width", 1)
        )
        hw_opt_node.add_attribute(
            "memory_bandwidth", caps.get("memory_bandwidth", "medium")
        )
        hw_opt_node.add_attribute("cache_size", caps.get("cache_size", "medium"))

        # Add device-specific optimizations
        if caps.get("neon_simd"):
            hw_opt_node.add_attribute("neon_simd", True)
        if caps.get("dsp_instructions"):
            hw_opt_node.add_attribute("dsp_instructions", True)
        if caps.get("cuda_cores"):
            hw_opt_node.add_attribute("cuda_cores", caps["cuda_cores"])

        return hw_opt_node

    def get_name(self) -> str:
        return self.name


class OptimizationPipeline:
    """Pipeline for applying multiple optimization passes."""

    def __init__(self):
        self.passes: List[UIRTransformation] = []
        self.results: List[OptimizationResult] = []

    def add_pass(self, pass_instance: UIRTransformation) -> None:
        """Add an optimization pass to the pipeline."""
        self.passes.append(pass_instance)
        logger.info(f"Added optimization pass: {pass_instance.get_name()}")

    def apply_optimizations(
        self, graph: UIRGraph
    ) -> Tuple[UIRGraph, List[OptimizationResult]]:
        """Apply all optimization passes to the graph."""
        logger.info(f"Applying {len(self.passes)} optimization passes")

        current_graph = graph
        results = []

        for pass_instance in self.passes:
            try:
                logger.info(f"Applying pass: {pass_instance.get_name()}")
                current_graph = pass_instance.transform(current_graph)

                # Create success result
                result = OptimizationResult(
                    optimization_type=self._get_optimization_type(pass_instance),
                    success=True,
                    metrics={
                        "pass_name": pass_instance.get_name(),
                        "nodes_before": len(graph.nodes),
                        "nodes_after": len(current_graph.nodes),
                    },
                )
                results.append(result)

            except Exception as e:
                logger.error(
                    f"Optimization pass {pass_instance.get_name()} failed: {e}"
                )

                # Create error result
                result = OptimizationResult(
                    optimization_type=self._get_optimization_type(pass_instance),
                    success=False,
                    errors=[str(e)],
                    metrics={"pass_name": pass_instance.get_name()},
                )
                results.append(result)

        self.results = results
        return current_graph, results

    def _get_optimization_type(
        self, pass_instance: UIRTransformation
    ) -> OptimizationType:
        """Get optimization type from pass instance."""
        if isinstance(pass_instance, QuantizationPass):
            return OptimizationType.QUANTIZATION
        elif isinstance(pass_instance, PruningPass):
            return OptimizationType.PRUNING
        elif isinstance(pass_instance, FusionPass):
            return OptimizationType.FUSION
        elif isinstance(pass_instance, MemoryOptimizationPass):
            return OptimizationType.MEMORY_OPTIMIZATION
        elif isinstance(pass_instance, HardwareSpecificOptimizationPass):
            return OptimizationType.HARDWARE_SPECIFIC
        else:
            return OptimizationType.GRAPH_OPTIMIZATION

    def create_edge_optimization_pipeline(
        self,
        target_device: str = "cpu",
        quantization_type: QuantizationType = QuantizationType.INT8,
        pruning_sparsity: float = 0.5,
    ) -> "OptimizationPipeline":
        """Create a standard edge optimization pipeline."""
        pipeline = OptimizationPipeline()

        # Add standard optimization passes
        pipeline.add_pass(QuantizationPass(quantization_type))
        pipeline.add_pass(PruningPass(pruning_sparsity))
        pipeline.add_pass(FusionPass())
        pipeline.add_pass(MemoryOptimizationPass())
        pipeline.add_pass(HardwareSpecificOptimizationPass(target_device))

        return pipeline


def create_optimization_pipeline(
    target_device: str = "cpu",
    quantization_type: QuantizationType = QuantizationType.INT8,
    pruning_sparsity: float = 0.5,
) -> OptimizationPipeline:
    """Create a standard optimization pipeline.

    Args:
        target_device: Target deployment device
        quantization_type: Type of quantization to apply
        pruning_sparsity: Sparsity level for pruning

    Returns:
        OptimizationPipeline: Configured optimization pipeline
    """
    pipeline = OptimizationPipeline()

    # Add standard optimization passes
    pipeline.add_pass(QuantizationPass(quantization_type))
    pipeline.add_pass(PruningPass(pruning_sparsity))
    pipeline.add_pass(FusionPass())
    pipeline.add_pass(MemoryOptimizationPass())
    pipeline.add_pass(HardwareSpecificOptimizationPass(target_device))

    return pipeline


if __name__ == "__main__":
    # Test the optimization passes
    from unified_ir import create_uir_from_edgeflow_config

    test_config = {
        "model": "test_model.tflite",
        "quantize": "int8",
        "target_device": "raspberry_pi",
        "input_shape": "1,224,224,3",
    }

    # Create UIR graph
    graph = create_uir_from_edgeflow_config(test_config)
    print(f"Created UIR graph: {graph.name}")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Tensors: {len(graph.tensors)}")

    # Create optimization pipeline
    pipeline = create_optimization_pipeline("raspberry_pi", QuantizationType.INT8, 0.3)

    # Apply optimizations
    optimized_graph, results = pipeline.apply_optimizations(graph)

    print(f"\nOptimization Results:")
    print(f"  Optimized graph: {optimized_graph.name}")
    print(f"  Nodes: {len(optimized_graph.nodes)}")
    print(f"  Tensors: {len(optimized_graph.tensors)}")

    print(f"\nPass Results:")
    for result in results:
        print(
            f"  {result.optimization_type.value}: {'SUCCESS' if result.success else 'FAILED'}"
        )
        if result.errors:
            print(f"    Errors: {result.errors}")
        if result.metrics:
            print(f"    Metrics: {result.metrics}")

    print(f"\nOptimized Graph Metadata:")
    for key, value in optimized_graph.framework_metadata.items():
        print(f"  {key}: {value}")
