"""EdgeFlow Explainability Reporter

This module generates detailed explainability reports for EdgeFlow optimizations,
providing insights into why specific optimizations were applied and their expected impact.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class OptimizationExplanation:
    """Represents an explanation for a specific optimization decision."""

    def __init__(self, name: str, description: str, impact: str, reasoning: str):
        self.name = name
        self.description = description
        self.impact = impact
        self.reasoning = reasoning


class EdgeFlowExplainabilityReporter:
    """Generates detailed explainability reports for EdgeFlow optimizations."""

    def __init__(self):
        self.optimization_explanations = {
            "int8_quantization": OptimizationExplanation(
                name="INT8 Quantization",
                description="Converts 32-bit floating point weights and activations to 8-bit integers",
                impact="Reduces model size by ~75% and improves inference speed by 2-4x on supported hardware",
                reasoning="Quantization reduces memory bandwidth requirements and enables faster integer arithmetic operations",
            ),
            "float16_quantization": OptimizationExplanation(
                name="FLOAT16 Quantization",
                description="Converts 32-bit floating point to 16-bit floating point precision",
                impact="Reduces model size by ~50% with minimal accuracy loss on modern hardware",
                reasoning="FLOAT16 provides good balance between size reduction and numerical precision",
            ),
            "structured_pruning": OptimizationExplanation(
                name="Structured Pruning",
                description="Removes entire filters/channels from convolutional layers",
                impact="Reduces model size by 10-50% depending on sparsity, maintains hardware efficiency",
                reasoning="Structured pruning preserves computational efficiency by removing complete operations",
            ),
            "operator_fusion": OptimizationExplanation(
                name="Operator Fusion",
                description="Combines multiple operations into single optimized kernels",
                impact="Reduces memory access and improves cache locality, typically 10-30% speedup",
                reasoning="Fused operations reduce intermediate memory allocations and improve data locality",
            ),
            "conv_batchnorm_fusion": OptimizationExplanation(
                name="Conv-BatchNorm Fusion",
                description="Merges convolution and batch normalization operations",
                impact="Eliminates batch norm overhead during inference, 5-15% speedup",
                reasoning="Batch normalization can be folded into convolution weights during inference",
            ),
            "activation_fusion": OptimizationExplanation(
                name="Activation Fusion",
                description="Integrates activation functions into preceding operations",
                impact="Reduces kernel launches and memory transfers, 5-10% speedup",
                reasoning="Activation functions can be computed inline without separate kernel launches",
            ),
            "raspberry_pi_optimizations": OptimizationExplanation(
                name="Raspberry Pi Optimizations",
                description="Device-specific optimizations for ARM Cortex-A processors",
                impact="Optimizes for ARM NEON instructions and memory hierarchy",
                reasoning="ARM processors benefit from specific instruction patterns and cache optimizations",
            ),
            "cortex_m4_optimizations": OptimizationExplanation(
                name="Cortex-M4 Optimizations",
                description="Optimizations for ARM Cortex-M4 microcontrollers",
                impact="Optimizes for tight memory constraints and integer-only operations",
                reasoning="Cortex-M4 has limited memory and no floating-point unit, requiring integer-only optimizations",
            ),
        }

    def generate_explainability_report(
        self,
        config: Dict[str, Any],
        optimization_results: Dict[str, Any],
        ir_transformations: Dict[str, Any],
        benchmark_comparison: Dict[str, Any],
    ) -> str:
        """Generate a comprehensive explainability report.

        Args:
            config: Original EdgeFlow configuration
            optimization_results: Results from the optimization process
            ir_transformations: IR transformation results
            benchmark_comparison: Before/after benchmark comparison

        Returns:
            Detailed explainability report as markdown
        """
        report_sections = []

        # Header
        report_sections.append(self._generate_header(config))

        # Configuration analysis
        report_sections.append(self._analyze_configuration(config))

        # Optimization explanations
        report_sections.append(
            self._explain_optimizations(config, optimization_results)
        )

        # IR transformation insights
        report_sections.append(self._explain_ir_transformations(ir_transformations))

        # Performance impact analysis
        report_sections.append(self._analyze_performance_impact(benchmark_comparison))

        # Device-specific insights
        report_sections.append(self._explain_device_optimizations(config))

        # Recommendations
        report_sections.append(
            self._generate_recommendations(config, optimization_results)
        )

        return "\n\n".join(report_sections)

    def _generate_header(self, config: Dict[str, Any]) -> str:
        """Generate report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        device = config.get("target_device", "cpu")

        return f"""# EdgeFlow Optimization Explainability Report

**Generated:** {timestamp}  
**Target Device:** {device}  
**Model:** {config.get('model', 'Unknown')}

## Executive Summary

This report provides detailed explanations for the optimizations applied by EdgeFlow,
including the reasoning behind each optimization decision and its expected impact on
your specific deployment scenario."""

    def _analyze_configuration(self, config: Dict[str, Any]) -> str:
        """Analyze the input configuration and explain the optimization strategy."""
        device = config.get("target_device", "cpu")
        quantize = config.get("quantize", "none")
        optimize_for = config.get("optimize_for", "balanced")
        memory_limit = config.get("memory_limit")

        analysis = ["## Configuration Analysis\n"]

        # Device analysis
        device_constraints = {
            "raspberry_pi": "ARM Cortex-A processor with limited memory bandwidth",
            "jetson_nano": "ARM Cortex-A57 with integrated GPU, good for parallel workloads",
            "jetson_xavier": "ARM Cortex-A78AE with powerful GPU, suitable for complex models",
            "cortex_m4": "ARM Cortex-M4 microcontroller with tight memory constraints",
            "cortex_m7": "ARM Cortex-M7 microcontroller with DSP capabilities",
            "cpu": "Generic CPU target, optimizations focus on general efficiency",
            "gpu": "GPU target, optimizations focus on parallel processing efficiency",
        }

        analysis.append(f"**Target Device:** {device}")
        analysis.append(
            f"- {device_constraints.get(device, 'Unknown device characteristics')}"
        )

        # Optimization strategy
        analysis.append(f"\n**Optimization Goal:** {optimize_for}")
        if optimize_for == "latency":
            analysis.append(
                "- Focus: Minimize inference time through quantization and fusion"
            )
        elif optimize_for == "memory":
            analysis.append(
                "- Focus: Minimize memory usage through aggressive quantization and pruning"
            )
        elif optimize_for == "size":
            analysis.append(
                "- Focus: Minimize model size through quantization and structured pruning"
            )
        else:
            analysis.append("- Focus: Balanced optimization across all metrics")

        # Memory constraints
        if memory_limit:
            analysis.append(f"\n**Memory Constraint:** {memory_limit}MB")
            analysis.append("- Optimizations will be constrained by available memory")

        return "\n".join(analysis)

    def _explain_optimizations(
        self, config: Dict[str, Any], results: Dict[str, Any]
    ) -> str:
        """Explain the applied optimizations."""
        explanations = ["## Applied Optimizations\n"]

        optimizations_applied = results.get("optimizations_applied", [])

        if not optimizations_applied:
            explanations.append("No specific optimizations were applied.")
            return "\n".join(explanations)

        explanations.append(
            "The following optimizations were applied based on your configuration:\n"
        )

        for opt in optimizations_applied:
            if opt in self.optimization_explanations:
                exp = self.optimization_explanations[opt]
                explanations.append(f"### {exp.name}")
                explanations.append(f"**Description:** {exp.description}")
                explanations.append(f"**Expected Impact:** {exp.impact}")
                explanations.append(f"**Reasoning:** {exp.reasoning}\n")
            else:
                explanations.append(f"### {opt.replace('_', ' ').title()}")
                explanations.append("Custom optimization applied.\n")

        return "\n".join(explanations)

    def _explain_ir_transformations(self, ir_transformations: Dict[str, Any]) -> str:
        """Explain IR transformations."""
        explanations = ["## Intermediate Representation Transformations\n"]

        passes_applied = ir_transformations.get("passes_applied", 0)
        transformations = ir_transformations.get("transformations", [])

        explanations.append(f"**IR Passes Applied:** {passes_applied}")

        if transformations:
            explanations.append("\n**Transformations:**")
            for transform in transformations:
                explanations.append(f"- {transform.replace('_', ' ').title()}")

        nodes = ir_transformations.get("nodes", 0)
        edges = ir_transformations.get("edges", 0)
        explanations.append(f"\n**IR Graph:** {nodes} nodes, {edges} edges")

        is_valid = ir_transformations.get("is_valid", True)
        if is_valid:
            explanations.append("\n✅ IR graph validation passed")
        else:
            errors = ir_transformations.get("validation_errors", [])
            explanations.append(f"\n⚠️ IR graph validation failed: {', '.join(errors)}")

        return "\n".join(explanations)

    def _analyze_performance_impact(self, comparison: Dict[str, Any]) -> str:
        """Analyze the performance impact of optimizations."""
        analysis = ["## Performance Impact Analysis\n"]

        improvements = comparison.get("improvements", {})

        # Size reduction
        size_reduction = improvements.get("size_reduction_percent", 0.0)
        if size_reduction > 0:
            analysis.append(f"**Model Size Reduction:** {size_reduction:.1f}%")
            if size_reduction > 50:
                analysis.append(
                    "- Excellent size reduction achieved through aggressive optimization"
                )
            elif size_reduction > 25:
                analysis.append(
                    "- Good size reduction, model significantly more deployable"
                )
            else:
                analysis.append(
                    "- Modest size reduction, consider additional optimizations"
                )

        # Latency improvement
        latency_improvement = improvements.get("latency_improvement_percent", 0.0)
        if latency_improvement > 0:
            analysis.append(f"\n**Latency Improvement:** {latency_improvement:.1f}%")
            if latency_improvement > 100:
                analysis.append("- Exceptional latency improvement, model much faster")
            elif latency_improvement > 50:
                analysis.append("- Significant latency improvement achieved")
            else:
                analysis.append("- Modest latency improvement")

        # Memory improvement
        memory_improvement = improvements.get("memory_improvement_percent", 0.0)
        if memory_improvement > 0:
            analysis.append(f"\n**Memory Usage Reduction:** {memory_improvement:.1f}%")
            analysis.append(
                "- Reduced memory footprint enables deployment on constrained devices"
            )

        # Throughput improvement
        throughput_improvement = improvements.get("throughput_improvement_percent", 0.0)
        if throughput_improvement > 0:
            analysis.append(
                f"\n**Throughput Improvement:** {throughput_improvement:.1f}%"
            )
            analysis.append(
                "- Higher throughput enables processing more data per second"
            )

        return "\n".join(analysis)

    def _explain_device_optimizations(self, config: Dict[str, Any]) -> str:
        """Explain device-specific optimizations."""
        device = config.get("target_device", "cpu")

        explanations = ["## Device-Specific Optimizations\n"]

        device_explanations = {
            "raspberry_pi": [
                "**ARM NEON Optimization:** Vectorized operations for ARM NEON SIMD instructions",
                "**Memory Hierarchy:** Optimized for L1/L2 cache sizes and memory bandwidth",
                "**Power Efficiency:** Reduced memory access patterns to minimize power consumption",
            ],
            "jetson_nano": [
                "**GPU Acceleration:** Optimized for integrated Maxwell GPU",
                "**Unified Memory:** Leveraged shared CPU-GPU memory architecture",
                "**Parallel Processing:** Optimized for parallel workload distribution",
            ],
            "jetson_xavier": [
                "**Volta GPU:** Optimized for powerful integrated GPU",
                "**High Bandwidth:** Leveraged high-speed memory interfaces",
                "**Multi-Core:** Optimized for multiple ARM Cortex-A78AE cores",
            ],
            "cortex_m4": [
                "**Integer-Only:** Converted to integer-only operations (no FPU)",
                "**Memory Constraints:** Optimized for tight SRAM limitations",
                "**Real-Time:** Optimized for deterministic, real-time performance",
            ],
            "cortex_m7": [
                "**DSP Optimization:** Leveraged ARM Cortex-M7 DSP instructions",
                "**Cache Optimization:** Optimized for instruction and data caches",
                "**Floating Point:** Used single-precision FPU when available",
            ],
        }

        if device in device_explanations:
            explanations.append(
                f"**{device.replace('_', ' ').title()} Optimizations:**"
            )
            for explanation in device_explanations[device]:
                explanations.append(f"- {explanation}")
        else:
            explanations.append("Generic CPU optimizations applied.")

        return "\n".join(explanations)

    def _generate_recommendations(
        self, config: Dict[str, Any], results: Dict[str, Any]
    ) -> str:
        """Generate optimization recommendations."""
        recommendations = ["## Optimization Recommendations\n"]

        device = config.get("target_device", "cpu")
        quantize = config.get("quantize", "none")
        memory_limit = config.get("memory_limit")

        # Quantization recommendations
        if quantize == "none":
            recommendations.append("### Quantization Opportunities")
            recommendations.append(
                "- Consider INT8 quantization for significant size and speed improvements"
            )
            if device in ["jetson_nano", "jetson_xavier"]:
                recommendations.append(
                    "- FLOAT16 quantization may provide good balance on GPU-enabled devices"
                )

        # Device-specific recommendations
        if device == "raspberry_pi":
            recommendations.append("\n### Raspberry Pi Specific")
            recommendations.append(
                "- Use structured pruning to maintain hardware efficiency"
            )
            recommendations.append(
                "- Consider reducing model complexity if memory is constrained"
            )

        elif device in ["cortex_m4", "cortex_m7"]:
            recommendations.append("\n### Microcontroller Specific")
            recommendations.append(
                "- Ensure all operations are integer-only for Cortex-M4"
            )
            recommendations.append(
                "- Consider model compression techniques for tight memory constraints"
            )

        # Memory recommendations
        if memory_limit and memory_limit < 256:
            recommendations.append("\n### Memory Optimization")
            recommendations.append(
                "- Consider more aggressive pruning for very tight memory constraints"
            )
            recommendations.append("- Evaluate if model complexity can be reduced")

        # Performance recommendations
        size_reduction = results.get("size_reduction_percent", 0.0)
        if size_reduction < 25:
            recommendations.append("\n### Further Optimization Potential")
            recommendations.append(
                "- Model may benefit from additional quantization or pruning"
            )
            recommendations.append("- Consider model architecture optimization")

        return "\n".join(recommendations)


def generate_explainability_report(
    config: Dict[str, Any],
    optimization_results: Dict[str, Any],
    ir_transformations: Dict[str, Any],
    benchmark_comparison: Dict[str, Any],
) -> str:
    """Generate a comprehensive explainability report.

    Args:
        config: Original EdgeFlow configuration
        optimization_results: Results from the optimization process
        ir_transformations: IR transformation results
        benchmark_comparison: Before/after benchmark comparison

    Returns:
        Detailed explainability report as markdown
    """
    reporter = EdgeFlowExplainabilityReporter()
    return reporter.generate_explainability_report(
        config, optimization_results, ir_transformations, benchmark_comparison
    )
