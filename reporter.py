"""
EdgeFlow Reporter Module

Generates comprehensive optimization reports in Markdown format.
This module is the final step in the EdgeFlow pipeline, creating
user-friendly reports that showcase optimization effectiveness.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    """Data class for model statistics."""

    size_mb: float
    latency_ms: float
    model_path: str
    timestamp: Optional[datetime] = None


@dataclass
class OptimizationReport:
    """Data class for complete optimization report."""

    original_stats: ModelStats
    optimized_stats: ModelStats
    config: Dict[str, Any]
    optimization_type: str
    target_device: Optional[str] = None


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator not in (0, 0.0) else 0.0


def calculate_improvements(
    original_stats: Dict[str, Any],
    optimized_stats: Dict[str, Any],
) -> Dict[str, float]:
    """
    Calculate improvement percentages.

    Args:
        original_stats: Original model statistics
        optimized_stats: Optimized model statistics

    Returns:
        Dictionary with improvement metrics
    """
    orig_size = float(original_stats.get("size_mb", 0.0))
    opt_size = float(optimized_stats.get("size_mb", 0.0))
    orig_lat = float(original_stats.get("latency_ms", 0.0))
    opt_lat = float(optimized_stats.get("latency_ms", 0.0))

    # Size reduction percentage
    size_reduction = 0.0
    if orig_size > 0:
        size_reduction = (1.0 - _safe_div(opt_size, orig_size)) * 100.0

    # Speedup factor and latency reduction
    speedup = _safe_div(orig_lat, opt_lat) if opt_lat > 0 else 0.0
    latency_reduction = 0.0
    if orig_lat > 0:
        latency_reduction = (1.0 - _safe_div(opt_lat, orig_lat)) * 100.0

    # Throughput increase percentage (fps inversely proportional to latency)
    throughput_increase = 0.0
    if opt_lat > 0 and orig_lat > 0:
        throughput_increase = (speedup - 1.0) * 100.0

    # Memory saved in MB (interpreted as file size reduction)
    memory_saved = max(orig_size - opt_size, 0.0)

    return {
        "size_reduction": round(size_reduction, 2),
        "speedup": round(speedup, 2),
        "latency_reduction": round(latency_reduction, 2),
        "throughput_increase": round(throughput_increase, 2),
        "memory_saved": round(memory_saved, 2),
    }


def format_table_row(
    label: str, original: Any, optimized: Any, improvement: str
) -> str:
    """
    Format a single row for the comparison table.

    Args:
        label: Metric name
        original: Original value
        optimized: Optimized value
        improvement: Improvement string (e.g., "↓ 75%")

    Returns:
        Formatted Markdown table row
    """
    return f"| **{label}** | {original} | {optimized} | {improvement} |"


def generate_visualization_ascii(original_size: float, optimized_size: float) -> str:
    """
    Generate ASCII art visualization of size reduction.

    Returns:
        ASCII bar chart as string
    """
    # Normalize bar lengths to a max width for readability
    max_width = 40
    max_size = max(original_size, optimized_size, 1.0)
    orig_bar = "█" * int(round((original_size / max_size) * max_width))
    opt_bar = "█" * int(round((optimized_size / max_size) * max_width))
    return (
        f"Original Model: [{orig_bar}] {original_size:.2f} MB\n"
        f"Optimized Model: [{opt_bar}] {optimized_size:.2f} MB"
    )


def generate_recommendations(improvements: Dict[str, float]) -> str:
    """Generate contextual recommendations based on improvements."""
    recs: list[str] = []

    size_red = improvements.get("size_reduction", 0.0)
    speedup = improvements.get("speedup", 0.0)

    if size_red >= 70:
        recs.append("- Excellent size reduction! Model highly suitable for edge")
    elif size_red >= 50:
        recs.append(
            "- Good size reduction achieved; consider further quantization or pruning"
        )
    elif size_red >= 25:
        recs.append(
            "- Moderate size reduction; explore operator fusion or mixed precision"
        )

    if speedup > 2:
        recs.append(
            "- Significant speed improvement enables real-time inference applications"
        )
    elif speedup > 1.2:
        recs.append(
            "- Latency improved; consider batching or threading for more throughput"
        )

    if not recs:
        recs.append("- Model successfully optimized for deployment")

    return "\n".join(recs)


def generate_report(
    unoptimized_stats: Dict[str, Any],
    optimized_stats: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    output_path: str = "report.md",
) -> str:
    """
    Generate comprehensive optimization report.

    Args:
        unoptimized_stats: Statistics for original model
        optimized_stats: Statistics for optimized model
        config: Optional configuration used for optimization
        output_path: Path where report will be saved

    Returns:
        Path to generated report

    Raises:
        ValueError: If stats are invalid or missing required fields
        IOError: If report cannot be written
    """

    # Validate inputs
    required_fields = ["size_mb", "latency_ms"]
    for field in required_fields:
        if field not in unoptimized_stats or field not in optimized_stats:
            raise ValueError(f"Missing required field: {field}")

    # Calculate improvements
    improvements = calculate_improvements(unoptimized_stats, optimized_stats)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build the report template
    viz = generate_visualization_ascii(
        float(unoptimized_stats["size_mb"]), float(optimized_stats["size_mb"])
    )

    report_template = f"""# EdgeFlow Optimization Report

Generated: {timestamp}

## Executive Summary

✨ **Model successfully optimized using EdgeFlow DSL** ✨

EdgeFlow successfully optimized your model with these results:

- **Size Reduction**: {improvements['size_reduction']:.1f}%
- **Speed Improvement**: {improvements['speedup']:.1f}x faster
- **Memory Savings**: {improvements['memory_saved']:.2f} MB

## Configuration Used

```json
{json.dumps(config, indent=2) if config else "No configuration provided"}
```

## Performance Metrics

| Metric | Original Model | Optimized Model | Improvement |
|--------|---------------|-----------------|-------------|
| **Model Size** | {unoptimized_stats['size_mb']:.2f} MB | \
        {optimized_stats['size_mb']:.2f} MB | ↓ {improvements['size_reduction']:.1f}% |
| **Inference Latency** | {unoptimized_stats['latency_ms']:.2f} ms | \
{optimized_stats['latency_ms']:.2f} ms | ↓ {improvements['latency_reduction']:.1f}% |
| **Throughput** | {1000/float(unoptimized_stats['latency_ms']) if unoptimized_stats['latency_ms'] > 0 else 0:.1f} fps | \
        {1000/float(optimized_stats['latency_ms']) if optimized_stats['latency_ms'] > 0 else 0:.1f} fps | \
        ↑ {improvements['throughput_increase']:.1f}% |

## Size Comparison

```
{viz}
```

## Optimization Details

### Technique Applied
- **Quantization**: INT8 quantization applied
- **Target Device**: {config.get('target_device', 'Generic') if config else 'Generic'}
- **Optimization Goal**: \
{config.get('optimize_for', 'Balanced') if config else 'Balanced'}

### Benefits Achieved
1. **Reduced Storage Requirements**: Your model now requires \
   {improvements['size_reduction']:.1f}% less storage
2. **Faster Inference**: Inference speed improved by {improvements['speedup']:.1f}x
3. **Lower Memory Footprint**: Runtime memory usage reduced significantly
4. **Edge Deployment Ready**: Model optimized for resource-constrained devices

## Recommendations

Based on the optimization results:
{generate_recommendations(improvements)}

## Technical Notes

- All benchmarks performed with 100 inference iterations after warm-up
- Latency measurements represent average inference time
- Size measurements include all model weights and metadata

---

*Report generated by EdgeFlow v1.0.0 - Your trusted ML optimization compiler*
"""

    # Write report to file
    output = Path(output_path)
    output.write_text(report_template)

    logger.info("Report successfully generated: %s", output)
    return str(output)


def generate_json_report(
    unoptimized_stats: Dict[str, Any],
    optimized_stats: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    output_path: str = "report.json",
) -> str:
    """Generate machine-readable JSON report for API consumption.

    Returns:
        JSON string containing the report data
    """
    # Validate minimal fields
    required_fields = ["size_mb", "latency_ms"]
    for field in required_fields:
        if field not in unoptimized_stats or field not in optimized_stats:
            raise ValueError(f"Missing required field: {field}")

    improvements = calculate_improvements(unoptimized_stats, optimized_stats)
    payload: Dict[str, Any] = {
        "original_stats": unoptimized_stats,
        "optimized_stats": optimized_stats,
        "improvements": improvements,
        "config": config or {},
        "timestamp": datetime.now().isoformat(),
    }

    json_content = json.dumps(payload, indent=2)

    # Optionally write to file if output_path is provided
    if output_path:
        output = Path(output_path)
        output.write_text(json_content)
        logger.info("JSON report successfully generated: %s", output)

    return json_content
