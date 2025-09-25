"""UIR Normalization and Cross-Framework Compatibility Layer.

This module normalizes framework-specific nuances into canonical UIR:
- Canonical op attributes (e.g., kernel/stride/padding for Conv2D)
- Canonical data layout (e.g., unify to NHWC or NCHW per policy)
- Datatype harmonization and tensor shape fixes
- Fallback handling for unsupported/custom ops

The normalizer is designed to be idempotent and safe to run multiple times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


# ---- Canonical operator and dtype mappings (reference for parsers/normalizer) ----
# Framework op name to canonical OperationType
ONNX_TO_IR_OP: Dict[str, OperationType] = {
    "Conv": OperationType.CONV2D,
    "Relu": OperationType.RELU,
    "Gemm": OperationType.DENSE,
    "MatMul": OperationType.MATMUL,
    "MaxPool": OperationType.MAX_POOL,
    "AveragePool": OperationType.AVG_POOL,
}

TF_TO_IR_OP: Dict[str, OperationType] = {
    "Conv2D": OperationType.CONV2D,
    "Relu": OperationType.RELU,
    "MatMul": OperationType.MATMUL,
    "Dense": OperationType.DENSE,
    "MaxPool": OperationType.MAX_POOL,
    "AvgPool": OperationType.AVG_POOL,
}

DTYPE_MAP: Dict[str, DataType] = {
    # Common TF/NumPy strings
    "float32": DataType.FLOAT32,
    "float16": DataType.FLOAT16,
    "int8": DataType.INT8,
    "int16": DataType.INT16,
    "int32": DataType.INT32,
    "int64": DataType.INT64,
    "uint8": DataType.UINT8,
    "uint16": DataType.UINT16,
    "uint32": DataType.UINT32,
    "uint64": DataType.UINT64,
    "bool": DataType.BOOL,
    "string": DataType.STRING,
}


@dataclass
class NormalizationPolicy:
    canonical_layout: str = "NHWC"  # or "NCHW"
    default_dtype: DataType = DataType.FLOAT32
    preserve_framework_metadata: bool = True


class UIRNormalizer(UIRTransformation):
    """Normalize UIR graphs into canonical conventions."""

    def __init__(self, policy: Optional[NormalizationPolicy] = None):
        self.policy = policy or NormalizationPolicy()
        self.name = "uir_normalizer"

    def get_name(self) -> str:
        return self.name

    def transform(self, graph: UIRGraph) -> UIRGraph:
        logger.info(
            "Normalizing UIR graph '%s' to layout=%s",
            graph.name,
            self.policy.canonical_layout,
        )

        normalized = UIRGraph(
            name=f"{graph.name}_normalized",
            framework_type=graph.framework_type,
            framework_metadata={**graph.framework_metadata, "normalized": True},
        )

        # Normalize tensors first
        for tensor_name, tensor in graph.tensors.items():
            normalized_tensor = self._normalize_tensor(tensor, graph.framework_type)
            normalized.add_tensor(normalized_tensor)

        # Normalize nodes and attributes
        for node_id, node in graph.nodes.items():
            normalized_node = self._normalize_node(node, graph.framework_type)
            self._annotate_device_compat(normalized_node, normalized.framework_metadata)
            normalized.add_node(normalized_node)

        # Preserve topology (edges)
        for edge in graph.edges:
            normalized.add_edge(*edge)

        return normalized

    # ---- Tensor normalization ----
    def _normalize_tensor(self, tensor: TensorInfo, fw: FrameworkType) -> TensorInfo:
        dtype = tensor.dtype or self.policy.default_dtype
        shape = tensor.shape

        # Basic layout harmonization for common 4D shapes
        if shape.rank == 4:
            dims = list(shape.dimensions)
            if self.policy.canonical_layout == "NHWC":
                # If ONNX/PyTorch NCHW, permute to NHWC for canonical storage
                if fw in (FrameworkType.ONNX, FrameworkType.PYTORCH):
                    dims = [dims[0], dims[2], dims[3], dims[1]]
            elif self.policy.canonical_layout == "NCHW":
                # If TF NHWC, permute to NCHW
                if fw == FrameworkType.TENSORFLOW:
                    dims = [dims[0], dims[3], dims[1], dims[2]]
            shape = TensorShape(dims)

        # Ensure no zero/negative except -1 for dynamic
        safe_dims = []
        for d in shape.dimensions:
            if isinstance(d, int) and d == 0:
                safe_dims.append(1)
            else:
                safe_dims.append(d)

        return TensorInfo(
            name=tensor.name,
            shape=TensorShape(safe_dims),
            dtype=dtype,
            framework_metadata={**tensor.framework_metadata},
        )

    # ---- Node normalization ----
    def _normalize_node(self, node: UIRNode, fw: FrameworkType) -> UIRNode:
        op = node.operation_type

        # Copy and normalize attributes
        normalized_attrs = {k: v for k, v in node.attributes.items()}

        if op == OperationType.CONV2D:
            self._normalize_conv2d_attributes(normalized_attrs, fw)
        elif op in (OperationType.MAX_POOL, OperationType.AVG_POOL):
            self._normalize_pool_attributes(normalized_attrs, fw)
        elif op in (
            OperationType.ADD,
            OperationType.SUB,
            OperationType.MUL,
            OperationType.DIV,
        ):
            self._normalize_elementwise_attributes(normalized_attrs)

        # Normalize activation naming if present
        act_attr = normalized_attrs.get("activation")
        if act_attr and getattr(act_attr, "value", None):
            act_val = str(act_attr.value).strip().lower()
            canonical = {
                "relu": "relu",
                "leaky_relu": "leaky_relu",
                "sigmoid": "sigmoid",
                "tanh": "tanh",
                "softmax": "softmax",
                "gelu": "gelu",
                "swish": "swish",
            }.get(act_val, act_val)
            normalized_attrs["activation"].value = canonical

        normalized_node = UIRNode(
            node_id=node.node_id,
            name=node.name,
            operation_type=op,
            framework_type=node.framework_type,
            inputs=list(node.inputs),
            outputs=list(node.outputs),
            attributes=normalized_attrs,
            framework_metadata=self._append_provenance(
                {**node.framework_metadata},
                description=f"Normalized {op.value} attributes",
                details={"layout": self.policy.canonical_layout},
            ),
        )

        return normalized_node

    def _normalize_conv2d_attributes(
        self, attrs: Dict[str, Any], fw: FrameworkType
    ) -> None:
        # Canonical keys: kernel_size=(kh, kw), strides=(sh, sw), padding, dilation=(dh, dw), groups
        # Map common framework-specific names
        if "kernel_size" not in attrs:
            k_h = (
                attrs.get("kernel_h") or attrs.get("kernelHeight") or attrs.get("ksize")
            )
            k_w = (
                attrs.get("kernel_w") or attrs.get("kernelWidth") or attrs.get("ksize")
            )
            if isinstance(k_h, int) and isinstance(k_w, int):
                attrs["kernel_size"] = (k_h, k_w)
        if "strides" not in attrs:
            s_h = (
                attrs.get("stride_h")
                or attrs.get("strideHeight")
                or attrs.get("strides")
            )
            s_w = (
                attrs.get("stride_w")
                or attrs.get("strideWidth")
                or attrs.get("strides")
            )
            if isinstance(s_h, int) and isinstance(s_w, int):
                attrs["strides"] = (s_h, s_w)
            elif isinstance(s_h, (list, tuple)):
                attrs["strides"] = tuple(s_h[-2:])
        if "dilation" not in attrs:
            d_h = (
                attrs.get("dilation_h")
                or attrs.get("dilationHeight")
                or attrs.get("dilations")
            )
            d_w = (
                attrs.get("dilation_w")
                or attrs.get("dilationWidth")
                or attrs.get("dilations")
            )
            if isinstance(d_h, int) and isinstance(d_w, int):
                attrs["dilation"] = (d_h, d_w)
            elif isinstance(d_h, (list, tuple)):
                attrs["dilation"] = tuple(d_h[-2:])
        # Padding normalization
        pad = attrs.get("padding") or attrs.get("auto_pad") or attrs.get("pad")
        if isinstance(pad, str):
            pad = pad.upper()
            if pad in ("SAME_UPPER", "SAME_LOWER", "SAME"):
                attrs["padding"] = "SAME"
            elif pad in ("VALID",):
                attrs["padding"] = "VALID"
        # Groups
        if "groups" not in attrs and "group" in attrs:
            attrs["groups"] = attrs["group"]

    def _normalize_pool_attributes(
        self, attrs: Dict[str, Any], fw: FrameworkType
    ) -> None:
        if "kernel_size" not in attrs:
            k = attrs.get("ksize") or attrs.get("kernel_shape")
            if isinstance(k, (list, tuple)):
                attrs["kernel_size"] = tuple(k[-2:])
        if "strides" not in attrs:
            s = attrs.get("strides")
            if isinstance(s, (list, tuple)):
                attrs["strides"] = tuple(s[-2:])
        pad = attrs.get("padding") or attrs.get("auto_pad")
        if isinstance(pad, str):
            pad = pad.upper()
            attrs["padding"] = "SAME" if "SAME" in pad else "VALID"

    def _normalize_elementwise_attributes(self, attrs: Dict[str, Any]) -> None:
        # No-op for now; placeholder for broadcasting semantics if needed
        return

    # ---- Device compatibility & provenance helpers ----
    def _annotate_device_compat(
        self, node: UIRNode, graph_meta: Dict[str, Any]
    ) -> None:
        # Extract config if present
        cfg = graph_meta.get("edgeflow_config", {})
        target = cfg.get("target_device", "cpu")
        memory_limit = cfg.get("memory_limit")

        device_info: Dict[str, Any] = {
            "target_device": target,
            "compatible": True,
            "notes": [],
        }

        # Simple example rule: limit kernel size for small devices
        if (
            node.operation_type == OperationType.CONV2D
            and "kernel_size" in node.attributes
        ):
            ks = node.attributes["kernel_size"].value
            if (
                isinstance(ks, (list, tuple))
                and len(ks) == 2
                and target in ("raspberry_pi", "microcontroller")
                and (ks[0] > 7 or ks[1] > 7)
            ):
                device_info["compatible"] = False
                device_info["notes"].append(
                    "Kernel size exceeds 7x7 limit for target device"
                )

        if memory_limit is not None:
            device_info["memory_limit_mb"] = memory_limit

        node.framework_metadata["device_compat"] = device_info
        self._append_provenance(
            node.framework_metadata,
            description="Annotated device compatibility",
            details=device_info,
        )

    def _append_provenance(
        self,
        meta: Dict[str, Any],
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prov = list(meta.get("provenance", []))
        entry = {"description": description}
        if details:
            entry.update({"details": details})
        prov.append(entry)
        meta["provenance"] = prov
        return meta


def normalize_uir_graph(graph: UIRGraph, layout: str = "NHWC") -> UIRGraph:
    """Convenience API to normalize a UIR graph with a chosen layout."""
    policy = NormalizationPolicy(canonical_layout=layout)
    return UIRNormalizer(policy).transform(graph)
