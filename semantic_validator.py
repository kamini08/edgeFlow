"""Semantic validation passes for EdgeFlow IR/config.

This module provides semantic checks at two levels:
- Configuration-level validation for quick feedback.
- IR-level validation (graph semantics, shapes, dtypes, device support).

IR validation supports both the lightweight `edgeflow_ir.IRGraph` and the
unified IR in `unified_ir.UIRGraph`. The checks are designed to be resilient to
partial metadata; when information is missing, validators emit warnings instead
of failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Optional imports for IR types; guarded to avoid circular deps in tests
try:  # Lightweight IR
    from edgeflow_ir import IRGraph, IRNode  # type: ignore
except Exception:  # noqa: BLE001
    IRGraph = Any  # type: ignore
    IRNode = Any  # type: ignore

try:  # Unified IR
    from unified_ir import OperationType, UIRGraph, UIRNode  # type: ignore
except Exception:  # noqa: BLE001
    UIRGraph = Any  # type: ignore
    UIRNode = Any  # type: ignore
    OperationType = Any  # type: ignore


@dataclass
class Diagnostic:
    code: str
    severity: str  # error | warning | info
    message: str
    hint: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class SemanticValidator:
    def __init__(self, device_registry: Optional[Dict[str, Any]] = None) -> None:
        self.device_registry = device_registry or {}

    def validate_config(self, config: Dict[str, Any]) -> List[Diagnostic]:
        diagnostics: List[Diagnostic] = []

        # Example: simple rule stubs; extend with real IR-based checks later
        if "quantize" in config and config.get("quantize") == "int8":
            # Require a model present to be meaningful
            if "model" not in config:
                diagnostics.append(
                    Diagnostic(
                        code="EF101",
                        severity="error",
                        message="Quantization requires a model to be specified",
                        hint='Add `model = "path/to/file.tflite"`',
                        context={"parameter": "quantize"},
                    )
                )

        # Device presence
        if "target_device" in config:
            device = str(config["target_device"]).lower()
            if self.device_registry and device not in self.device_registry:
                diagnostics.append(
                    Diagnostic(
                        code="EF401",
                        severity="error",
                        message=f"Unsupported target device: {device}",
                        hint="Use a supported device or omit target_device",
                        context={"parameter": "target_device"},
                    )
                )

        return diagnostics

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
        cpu_caps = {
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
