"""EdgeFlow Model Benchmarker

Day 3/4 Implementation (Team B):
--------------------------------
Provides *real* model size and latency benchmarking when TensorFlow Lite is
available, with a graceful fallback to the earlier simulation-based metrics.

Public low-level functions now exposed for the pipeline:
    - ``get_model_size(model_path)``  -> float (MB)
    - ``benchmark_latency(model_path, runs=100, warmup=1)`` -> float (ms)

If TensorFlow (``tensorflow`` package) is not installed, or a model cannot be
loaded, the module silently falls back to deterministic simulation so tests
remain stable in lightweight environments (CI, dev containers, etc.).

Real benchmarking logic:
    * Loads the TFLite model with ``tf.lite.Interpreter``.
    * Allocates tensors and inspects the first input tensor to derive shape & dtype.
    * Performs one (configurable) warm-up inference.
    * Measures average latency across N runs using ``time.perf_counter``.
    * Computes an approximate throughput (FPS = 1000 / avg_latency_ms).

The higher-level convenience functions ``benchmark_model`` and
``compare_models`` retain their original signatures and output schema so the
rest of the codebase (and existing tests) continue to work unchanged.
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

try:  # Optional heavy dependency
    import tensorflow as _tf  # type: ignore

    _TF_AVAILABLE = True
except Exception:  # noqa: BLE001 - optional dependency
    _TF_AVAILABLE = False
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_model_size(model_path: str) -> float:
    """Return model size in megabytes.

    Args:
        model_path: Path to a file on disk
    Returns:
        Size in megabytes (float). Returns 0.0 if file missing.
    """
    try:
        return os.path.getsize(model_path) / (1024 * 1024)
    except OSError:
        return 0.0


def _generate_random_input(shape, dtype):  # type: ignore[no-untyped-def]
    """Generate random input tensor matching shape/dtype for benchmarking."""
    import numpy as np  # Local import to keep global namespace light

    if dtype == np.float32:
        return np.random.random(shape).astype(np.float32) * 1.0
    if dtype == np.int8:
        return np.random.randint(-128, 127, size=shape, dtype=np.int8)
    if dtype == np.uint8:
        return np.random.randint(0, 255, size=shape, dtype=np.uint8)
    # Fallback: float32
    return np.random.random(shape).astype(np.float32)


def benchmark_latency(
    model_path: str, runs: int = 100, warmup: int = 1
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Benchmark average inference latency (ms) for a TFLite model.

    Performs one or more warm-up runs followed by timed runs.

    Args:
        model_path: Path to a *.tflite model
        runs: Number of timed inference iterations
        warmup: Number of warm-up iterations (not timed)
    Returns:
        (avg_latency_ms, debug_metadata_dict_or_None)
        If TensorFlow Lite is unavailable or model can't be loaded, returns (0.0, None)
    """
    if not _TF_AVAILABLE or not os.path.isfile(model_path):
        return 0.0, None

    try:  # Load interpreter
        interpreter = _tf.lite.Interpreter(  # type: ignore[attr-defined]
            model_path=model_path
        )
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        if not input_details:
            return 0.0, None
        first_input = input_details[0]
        shape = first_input.get("shape")
        dtype = first_input.get("dtype")
        index = first_input.get("index")
        if shape is None or dtype is None or index is None:
            return 0.0, None

        # Warm-up
        for _ in range(max(warmup, 0)):
            data = _generate_random_input(shape, dtype)
            interpreter.set_tensor(index, data)
            interpreter.invoke()

        # Timed runs
        total = 0.0
        for _ in range(max(runs, 1)):
            data = _generate_random_input(shape, dtype)
            start = time.perf_counter()
            interpreter.set_tensor(index, data)
            interpreter.invoke()
            total += (time.perf_counter() - start) * 1000.0
        avg_ms = total / max(runs, 1)
        metadata = {
            "input_shape": tuple(int(x) for x in shape),
            "dtype": str(dtype),
            "runs": runs,
            "warmup": warmup,
        }
        return avg_ms, metadata
    except Exception:  # noqa: BLE001
        return 0.0, None


class EdgeFlowBenchmarker:
    """Comprehensive benchmarking for EdgeFlow models (real + simulated)."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize benchmarker with EdgeFlow configuration.

        Args:
            config: Parsed EdgeFlow configuration dictionary
        """
        self.config = config
        self.target_device = config.get("target_device", "cpu")
        self.optimize_for = config.get("optimize_for", "latency")
        self.memory_limit = float(config.get("memory_limit", 64))

    def benchmark_model(self, model_path: str) -> Dict[str, Any]:
        """Benchmark a single model (real if possible, else simulation).

        Args:
            model_path: Path to the model file

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking model: {model_path}")

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return self._create_dummy_benchmark(model_path)

        model_size_mb = get_model_size(model_path)

        # Attempt real latency measurement
        latency_ms, meta = benchmark_latency(model_path)
        used_real = latency_ms > 0.0

        if used_real:
            throughput_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            memory_usage_mb = round(min(model_size_mb * 2, self.memory_limit), 2)
            results = {
                "model_path": model_path,
                "model_size_mb": round(model_size_mb, 3),
                "device": self.target_device,
                "latency_ms": round(latency_ms, 2),
                "throughput_fps": round(throughput_fps, 2),
                "memory_usage_mb": memory_usage_mb,
                "optimize_for": self.optimize_for,
                "status": "success",
                "mode": "real",
            }
            if meta:
                results["details"] = meta
        else:
            # Simulation fallback
            results = self._simulate_benchmark(model_path, model_size_mb)
            results["mode"] = "simulation"

        logger.info(f"Benchmark complete: {model_path}")
        logger.info(
            f"  Latency: {results['latency_ms']:.2f} ms ({results.get('mode')})"
        )
        logger.info(f"  Throughput: {results['throughput_fps']:.2f} FPS")
        logger.info(f"  Memory: {results['memory_usage_mb']:.2f} MB")

        return results

    def compare_models(self, original_path: str, optimized_path: str) -> Dict[str, Any]:
        """Compare original and optimized models.

        Args:
            original_path: Path to original model
            optimized_path: Path to optimized model

        Returns:
            Dictionary with comparison results
        """
        logger.info("Running model comparison benchmark")

        # Benchmark both models
        original_results = self.benchmark_model(original_path)
        optimized_results = self.benchmark_model(optimized_path)

        # Calculate improvements
        improvements = self._calculate_improvements(original_results, optimized_results)

        comparison = {
            "original": original_results,
            "optimized": optimized_results,
            "improvements": improvements,
            "summary": self._generate_summary(improvements),
        }

        return comparison

    def _simulate_benchmark(
        self, model_path: str, model_size_mb: float
    ) -> Dict[str, Any]:
        """Simulate benchmarking based on device characteristics."""

        # Base performance characteristics by device
        device_specs = {
            "raspberry_pi": {
                "base_latency_ms": 50.0,
                "base_throughput_fps": 20.0,
                "base_memory_mb": 64.0,
                "size_factor": 0.8,  # Larger models are slower
            },
            "jetson_nano": {
                "base_latency_ms": 25.0,
                "base_throughput_fps": 40.0,
                "base_memory_mb": 128.0,
                "size_factor": 0.6,
            },
            "cpu": {
                "base_latency_ms": 10.0,
                "base_throughput_fps": 100.0,
                "base_memory_mb": 256.0,
                "size_factor": 0.4,
            },
        }

        specs = device_specs.get(self.target_device, device_specs["cpu"])

        # Calculate performance based on model size
        size_factor = 1 + (model_size_mb / 100) * specs["size_factor"]

        # Apply optimization goals
        if self.optimize_for == "latency":
            latency_multiplier = 0.7  # 30% faster
            throughput_multiplier = 1.2  # 20% more throughput
        elif self.optimize_for == "memory":
            latency_multiplier = 1.1  # 10% slower
            throughput_multiplier = 0.9  # 10% less throughput
        else:  # balanced
            latency_multiplier = 0.9  # 10% faster
            throughput_multiplier = 1.1  # 10% more throughput

        # Calculate final metrics
        latency_ms = specs["base_latency_ms"] * size_factor * latency_multiplier
        throughput_fps = (
            specs["base_throughput_fps"] / size_factor * throughput_multiplier
        )
        memory_usage_mb = min(specs["base_memory_mb"], model_size_mb * 2)

        return {
            "model_path": model_path,
            "model_size_mb": model_size_mb,
            "device": self.target_device,
            "latency_ms": round(latency_ms, 2),
            "throughput_fps": round(throughput_fps, 2),
            "memory_usage_mb": round(memory_usage_mb, 2),
            "optimize_for": self.optimize_for,
            "status": "success",
        }

    def _create_dummy_benchmark(self, model_path: str) -> Dict[str, Any]:
        """Create dummy benchmark results for non-existent model."""
        return {
            "model_path": model_path,
            "model_size_mb": 0.0,
            "device": self.target_device,
            "latency_ms": 0.0,
            "throughput_fps": 0.0,
            "memory_usage_mb": 0.0,
            "optimize_for": self.optimize_for,
            "status": "file_not_found",
        }

    def _calculate_improvements(
        self, original: Dict[str, Any], optimized: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate improvement metrics between original and optimized models."""

        # Handle missing models
        if original["status"] != "success" or optimized["status"] != "success":
            return {
                "size_reduction_percent": 0.0,
                "latency_improvement_percent": 0.0,
                "throughput_improvement_percent": 0.0,
                "memory_improvement_percent": 0.0,
            }

        # Calculate size reduction
        original_size = original["model_size_mb"]
        optimized_size = optimized["model_size_mb"]
        size_reduction = (
            ((original_size - optimized_size) / original_size * 100)
            if original_size > 0
            else 0.0
        )

        # Calculate latency improvement
        original_latency = original["latency_ms"]
        optimized_latency = optimized["latency_ms"]
        latency_improvement = (
            ((original_latency - optimized_latency) / original_latency * 100)
            if original_latency > 0
            else 0.0
        )

        # Calculate throughput improvement
        original_throughput = original["throughput_fps"]
        optimized_throughput = optimized["throughput_fps"]
        throughput_improvement = (
            ((optimized_throughput - original_throughput) / original_throughput * 100)
            if original_throughput > 0
            else 0.0
        )

        # Calculate memory improvement
        original_memory = original["memory_usage_mb"]
        optimized_memory = optimized["memory_usage_mb"]
        memory_improvement = (
            ((original_memory - optimized_memory) / original_memory * 100)
            if original_memory > 0
            else 0.0
        )

        return {
            "size_reduction_percent": round(size_reduction, 2),
            "latency_improvement_percent": round(latency_improvement, 2),
            "throughput_improvement_percent": round(throughput_improvement, 2),
            "memory_improvement_percent": round(memory_improvement, 2),
        }

    def _generate_summary(self, improvements: Dict[str, Any]) -> str:
        """Generate a human-readable summary of improvements."""
        summary_parts = []

        if improvements["size_reduction_percent"] > 10:
            summary_parts.append(
                f"Model size reduced by {improvements['size_reduction_percent']:.1f}%"
            )

        if improvements["latency_improvement_percent"] > 10:
            summary_parts.append(
                f"Inference {improvements['latency_improvement_percent']:.1f}% faster"
            )

        if improvements["throughput_improvement_percent"] > 10:
            summary_parts.append(
                f"Throughput improved by "
                f"{improvements['throughput_improvement_percent']:.1f}%"
            )
        elif improvements["throughput_improvement_percent"] < -10:
            summary_parts.append(
                f"Throughput decreased by "
                f"{abs(improvements['throughput_improvement_percent']):.1f}%"
            )

        return (
            "; ".join(summary_parts) if summary_parts else "No significant improvements"
        )


def benchmark_model(model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Benchmark wrapper retaining original public API."""
    """Benchmark a single model.

    Args:
        model_path: Path to the model file
        config: EdgeFlow configuration

    Returns:
        Benchmark results dictionary
    """
    benchmarker = EdgeFlowBenchmarker(config)
    return benchmarker.benchmark_model(model_path)


def compare_models(
    original_path: str, optimized_path: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare two models.

    Args:
        original_path: Path to original model
        optimized_path: Path to optimized model
        config: EdgeFlow configuration

    Returns:
        Comparison results dictionary
    """
    benchmarker = EdgeFlowBenchmarker(config)
    return benchmarker.compare_models(original_path, optimized_path)


__all__ = [
    "get_model_size",
    "benchmark_latency",
    "benchmark_model",
    "compare_models",
    "EdgeFlowBenchmarker",
]
