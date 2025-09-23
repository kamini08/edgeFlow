"""High-level pipeline API: import → normalize → validate → optimize → MLIR.

Provides a simple entrypoint to run the cross-framework workflow on a model path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from framework_parsers import parse_model_to_uir
from mlir_dialect import MLIRModule, MLIRPipeline, create_mlir_pipeline
from uir_normalizer import normalize_uir_graph
from uir_optimization_passes import QuantizationType, create_optimization_pipeline
from uir_validators import ValidationResult, validate_uir_graph
from unified_ir import UIRGraph

logger = logging.getLogger(__name__)


def compile_model(
    model_path: str,
    target_device: str = "cpu",
    quantize: str = "none",
    pruning_sparsity: float = 0.0,
    canonical_layout: str = "NHWC",
) -> Tuple[MLIRModule, UIRGraph, ValidationResult]:
    """Run the full pipeline on a model.

    Returns (mlir_module, final_graph, validation_result).
    """
    # 1) Import
    graph = parse_model_to_uir(model_path)

    # 2) Normalize
    graph = normalize_uir_graph(graph, layout=canonical_layout)

    # 3) Validate
    # Attach pipeline hints for downstream passes
    graph.framework_metadata.setdefault("target_device", target_device)
    graph.framework_metadata.setdefault("quantize", quantize)
    validation_result = validate_uir_graph(graph)

    # 4) Optimize
    q_type = (
        QuantizationType.INT8
        if quantize == "int8"
        else QuantizationType.FLOAT16
        if quantize == "float16"
        else QuantizationType.NONE
    )
    opt_pipeline = create_optimization_pipeline(
        target_device=target_device,
        quantization_type=q_type,
        pruning_sparsity=pruning_sparsity,
    )
    optimized_graph, _ = opt_pipeline.apply_optimizations(graph)

    # 5) MLIR
    mlir_pipeline: MLIRPipeline = create_mlir_pipeline(target_device)
    mlir_module, final_graph = mlir_pipeline.compile(optimized_graph, target_device)

    return mlir_module, final_graph, validation_result
