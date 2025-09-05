"""EdgeFlow benchmarker placeholder.

Provides a stub for benchmarking optimized models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def benchmark(model_path: str, device: str = "cpu") -> Dict[str, Any]:
    """Benchmark a TFLite model (placeholder).

    Args:
        model_path: Path to the model file to benchmark.
        device: Target device identifier (e.g., 'raspberry_pi').

    Returns:
        Dict[str, Any]: Minimal benchmark results.
    """

    logger.info(
        "[benchmarker] Benchmarking placeholder: model=%s on %s", model_path, device
    )
    return {
        "model_path": model_path,
        "device": device,
        "latency_ms": None,
        "throughput": None,
    }
