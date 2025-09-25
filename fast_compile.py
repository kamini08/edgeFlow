"""EdgeFlow Fast Compile System

This module implements fast compile-feedback cycles for rapid developer iteration,
providing immediate feedback on configuration changes without full optimization.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import our existing semantic analyzer components
from semantic_analyzer import SemanticAnalyzer, semantic_check
from semantic_analyzer.constraints import (
    ConstraintConfig,
    get_edge_device_config,
    get_mobile_device_config,
    get_server_device_config,
)
from semantic_analyzer.error_types import ErrorCollector, ErrorSeverity
from semantic_analyzer.ir_nodes import DataType, IRGraph, LayerType

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model configuration."""

    model_size_mb: float
    inference_time_ms: float
    memory_usage_mb: float
    power_consumption_mw: float
    accuracy_estimate: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class FastCompileResult:
    """Result from a fast compilation cycle."""

    success: bool
    errors: List[str]
    warnings: List[str]
    performance_metrics: Optional[PerformanceMetrics]
    compile_time_ms: float
    device_compatibility: Dict[str, bool]
    optimization_suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.performance_metrics:
            result["performance_metrics"] = self.performance_metrics.to_dict()
        return result


class PerformanceEstimator:
    """Dynamic performance estimation based on model characteristics and device profiles."""

    def __init__(self, device_profiles_path: Optional[str] = None):
        self.device_profiles = self._load_device_profiles(device_profiles_path)
        self.quantization_factors = self._initialize_quantization_factors()
        self.layer_complexity_factors = self._initialize_layer_complexity()

    def _load_device_profiles(
        self, profiles_path: Optional[str]
    ) -> Dict[str, Dict[str, float]]:
        """Load device performance profiles from file or use defaults."""
        if profiles_path and Path(profiles_path).exists():
            try:
                with open(profiles_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(
                    f"Failed to load device profiles from {profiles_path}: {e}"
                )

        # Dynamic device profiles based on known hardware characteristics
        return {
            "raspberry_pi_4": {
                "cpu_cores": 4,
                "cpu_freq_ghz": 1.5,
                "memory_gb": 4.0,
                "memory_bandwidth_gbps": 6.4,
                "power_budget_w": 15.0,
                "int8_ops_per_sec": 2.4e9,
                "float32_ops_per_sec": 0.6e9,
                "memory_efficiency": 0.8,
            },
            "jetson_nano": {
                "cpu_cores": 4,
                "cpu_freq_ghz": 1.43,
                "memory_gb": 4.0,
                "memory_bandwidth_gbps": 25.6,
                "power_budget_w": 10.0,
                "int8_ops_per_sec": 21e9,
                "float32_ops_per_sec": 0.472e12,
                "memory_efficiency": 1.2,
            },
            "jetson_xavier_nx": {
                "cpu_cores": 6,
                "cpu_freq_ghz": 1.9,
                "memory_gb": 8.0,
                "memory_bandwidth_gbps": 51.2,
                "power_budget_w": 20.0,
                "int8_ops_per_sec": 21e12,
                "float32_ops_per_sec": 1.37e12,
                "memory_efficiency": 1.5,
            },
            "cortex_m4": {
                "cpu_cores": 1,
                "cpu_freq_ghz": 0.168,
                "memory_gb": 0.001,  # 1MB
                "memory_bandwidth_gbps": 0.1,
                "power_budget_w": 0.1,
                "int8_ops_per_sec": 84e6,
                "float32_ops_per_sec": 21e6,
                "memory_efficiency": 0.6,
            },
            "intel_nuc": {
                "cpu_cores": 4,
                "cpu_freq_ghz": 2.7,
                "memory_gb": 16.0,
                "memory_bandwidth_gbps": 38.4,
                "power_budget_w": 65.0,
                "int8_ops_per_sec": 432e9,
                "float32_ops_per_sec": 108e9,
                "memory_efficiency": 1.0,
            },
        }

    def _initialize_quantization_factors(self) -> Dict[str, Dict[str, float]]:
        """Initialize quantization impact factors based on research data."""
        return {
            DataType.INT8.value: {
                "size_factor": 0.25,  # 4x smaller than float32
                "speed_factor": 2.0,  # 2x faster on average
                "memory_factor": 0.25,  # 4x less memory
                "accuracy_loss": 0.02,  # 2% accuracy loss typical
            },
            DataType.UINT8.value: {
                "size_factor": 0.25,
                "speed_factor": 2.2,  # Slightly faster than int8
                "memory_factor": 0.25,
                "accuracy_loss": 0.015,
            },
            DataType.FLOAT16.value: {
                "size_factor": 0.5,  # 2x smaller than float32
                "speed_factor": 1.3,  # 30% faster
                "memory_factor": 0.5,  # 2x less memory
                "accuracy_loss": 0.005,  # Minimal accuracy loss
            },
            DataType.FLOAT32.value: {
                "size_factor": 1.0,  # Baseline
                "speed_factor": 1.0,  # Baseline
                "memory_factor": 1.0,  # Baseline
                "accuracy_loss": 0.0,  # No loss
            },
        }

    def _initialize_layer_complexity(self) -> Dict[str, Dict[str, float]]:
        """Initialize computational complexity factors for different layer types."""
        return {
            LayerType.CONV2D.value: {
                "ops_per_param": 2.0,  # MAC operations per parameter
                "memory_overhead": 1.2,  # Additional memory for activations
                "parallelization": 0.8,  # How well it parallelizes
            },
            LayerType.DENSE.value: {
                "ops_per_param": 2.0,
                "memory_overhead": 1.1,
                "parallelization": 0.9,
            },
            LayerType.LSTM.value: {
                "ops_per_param": 8.0,  # LSTM is computationally intensive
                "memory_overhead": 2.0,  # Requires state storage
                "parallelization": 0.3,  # Sequential nature limits parallelization
            },
            LayerType.ATTENTION.value: {
                "ops_per_param": 4.0,
                "memory_overhead": 1.5,
                "parallelization": 0.7,
            },
            LayerType.BATCH_NORM.value: {
                "ops_per_param": 1.0,
                "memory_overhead": 1.0,
                "parallelization": 1.0,
            },
        }

    def estimate_performance(
        self, ir_graph: IRGraph, device_name: str, quantization: str = "float32"
    ) -> PerformanceMetrics:
        """Estimate performance metrics for a given model and device configuration."""
        device_profile = self.device_profiles.get(device_name)
        if not device_profile:
            logger.warning(f"Unknown device {device_name}, using default profile")
            device_profile = self.device_profiles["intel_nuc"]

        quant_factors = self.quantization_factors.get(
            quantization, self.quantization_factors[DataType.FLOAT32.value]
        )

        # Calculate model complexity
        total_params = 0
        total_ops = 0
        total_memory = 0
        accuracy_loss = quant_factors["accuracy_loss"]

        for node in ir_graph.nodes.values():
            layer_complexity = self.layer_complexity_factors.get(
                node.layer_type.value,
                self.layer_complexity_factors[LayerType.DENSE.value],
            )

            # Estimate parameters based on layer type and configuration
            node_params = self._estimate_layer_parameters(node)
            node_ops = node_params * layer_complexity["ops_per_param"]
            node_memory = node_params * layer_complexity["memory_overhead"]

            total_params += node_params
            total_ops += node_ops
            total_memory += node_memory

        # Apply quantization factors
        model_size_mb = (total_params * 4 * quant_factors["size_factor"]) / (
            1024 * 1024
        )
        memory_usage_mb = (total_memory * 4 * quant_factors["memory_factor"]) / (
            1024 * 1024
        )

        # Estimate inference time based on device capabilities
        ops_per_sec = device_profile.get(
            f"{quantization}_ops_per_sec", device_profile["float32_ops_per_sec"]
        )
        inference_time_ms = (
            (total_ops / ops_per_sec) * 1000 / quant_factors["speed_factor"]
        )

        # Estimate power consumption
        base_power = device_profile["power_budget_w"]
        utilization = min(
            1.0, total_ops / (ops_per_sec * 0.1)
        )  # Assume 10% baseline utilization
        power_consumption_mw = base_power * (0.3 + 0.7 * utilization) * 1000

        # Estimate accuracy (baseline 95% minus quantization loss)
        accuracy_estimate = max(0.0, 0.95 - accuracy_loss)

        return PerformanceMetrics(
            model_size_mb=model_size_mb,
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_usage_mb,
            power_consumption_mw=power_consumption_mw,
            accuracy_estimate=accuracy_estimate,
        )

    def _estimate_layer_parameters(self, node) -> int:
        """Estimate the number of parameters in a layer based on its configuration."""
        if node.layer_type == LayerType.CONV2D:
            # Conv2D: (kernel_h * kernel_w * input_channels + 1) * output_channels
            kernel_size = node.parameters.get("kernel_size", (3, 3))
            if isinstance(kernel_size, int):
                kernel_h = kernel_w = kernel_size
            else:
                kernel_h, kernel_w = kernel_size

            filters = node.parameters.get("filters", 32)
            input_channels = 3  # Estimate based on typical input

            return (kernel_h * kernel_w * input_channels + 1) * filters

        elif node.layer_type == LayerType.DENSE:
            # Dense: (input_size + 1) * output_size
            units = node.parameters.get("units", 128)
            input_size = 1024  # Estimate based on typical dense layer input

            return (input_size + 1) * units

        elif node.layer_type == LayerType.LSTM:
            # LSTM: 4 * (input_size + hidden_size + 1) * hidden_size
            units = node.parameters.get("units", 128)
            input_size = 100  # Estimate

            return 4 * (input_size + units + 1) * units

        # Default estimate for other layers
        return 1000


class EdgeFlowFastCompiler:
    """Provides fast compilation feedback for rapid development iteration."""

    def __init__(self, device_profiles_path: Optional[str] = None):
        self.performance_estimator = PerformanceEstimator(device_profiles_path)
        self.device_configs = {
            "edge": get_edge_device_config(),
            "mobile": get_mobile_device_config(),
            "server": get_server_device_config(),
        }

    def fast_compile(
        self,
        ir_graph: IRGraph,
        target_device: str = "mobile",
        quantization: str = "float32",
    ) -> FastCompileResult:
        """
        Perform fast compilation with immediate feedback.

        Args:
            ir_graph: The IR graph to compile
            target_device: Target device type ('edge', 'mobile', 'server')
            quantization: Quantization type ('int8', 'uint8', 'float16', 'float32')

        Returns:
            FastCompileResult with validation results and performance estimates
        """
        start_time = time.time()
        errors = []
        warnings = []
        optimization_suggestions = []

        try:
            # Step 1: Quick semantic validation
            device_config = self.device_configs.get(target_device)
            if not device_config:
                errors.append(f"Unknown target device: {target_device}")
                device_config = self.device_configs["mobile"]  # Fallback

            analyzer = SemanticAnalyzer(device_config)
            semantic_errors = analyzer.analyze(ir_graph)

            if semantic_errors.has_errors():
                errors.extend([str(e) for e in semantic_errors.errors])

            # Get warnings from the error collector
            warning_errors = semantic_errors.get_errors_by_severity(
                ErrorSeverity.WARNING
            )
            if warning_errors:
                warnings.extend([str(w) for w in warning_errors])

            # Step 2: Performance estimation
            device_name = self._map_device_type_to_name(target_device)
            performance_metrics = self.performance_estimator.estimate_performance(
                ir_graph, device_name, quantization
            )

            # Step 3: Device compatibility check
            device_compatibility = self._check_device_compatibility(ir_graph)

            # Step 4: Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                ir_graph, performance_metrics, target_device, quantization
            )

            compile_time_ms = (time.time() - start_time) * 1000

            return FastCompileResult(
                success=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                performance_metrics=performance_metrics,
                compile_time_ms=compile_time_ms,
                device_compatibility=device_compatibility,
                optimization_suggestions=optimization_suggestions,
            )

        except Exception as e:
            logger.error(f"Fast compilation failed: {e}")
            compile_time_ms = (time.time() - start_time) * 1000

            return FastCompileResult(
                success=False,
                errors=[f"Compilation error: {str(e)}"],
                warnings=warnings,
                performance_metrics=None,
                compile_time_ms=compile_time_ms,
                device_compatibility={},
                optimization_suggestions=[],
            )

    def _map_device_type_to_name(self, device_type: str) -> str:
        """Map device type to specific device name for performance estimation."""
        mapping = {
            "edge": "raspberry_pi_4",
            "mobile": "jetson_nano",
            "server": "intel_nuc",
        }
        return mapping.get(device_type, "intel_nuc")

    def _check_device_compatibility(self, ir_graph: IRGraph) -> Dict[str, bool]:
        """Check compatibility with different device types."""
        compatibility = {}

        for device_type, config in self.device_configs.items():
            analyzer = SemanticAnalyzer(config)
            errors = analyzer.analyze(ir_graph)
            compatibility[device_type] = not errors.has_fatal_errors()

        return compatibility

    def _generate_optimization_suggestions(
        self,
        ir_graph: IRGraph,
        performance_metrics: PerformanceMetrics,
        target_device: str,
        quantization: str,
    ) -> List[str]:
        """Generate optimization suggestions based on model analysis."""
        suggestions = []

        # Memory optimization suggestions
        if performance_metrics.memory_usage_mb > 100:
            suggestions.append(
                "Consider using int8 quantization to reduce memory usage by ~75%"
            )

        # Speed optimization suggestions
        if performance_metrics.inference_time_ms > 100:
            suggestions.append(
                "Model inference time is high - consider reducing model complexity"
            )
            if quantization == "float32":
                suggestions.append(
                    "Switch to float16 quantization for ~30% speed improvement"
                )

        # Power optimization suggestions
        if performance_metrics.power_consumption_mw > 5000:  # 5W
            suggestions.append(
                "High power consumption detected - consider model pruning"
            )

        # Device-specific suggestions
        if target_device == "edge":
            if performance_metrics.model_size_mb > 50:
                suggestions.append(
                    "Model size too large for edge deployment - consider distillation"
                )

            # Check for unsupported layer types
            unsupported_layers = []
            for node in ir_graph.nodes.values():
                if node.layer_type in [LayerType.LSTM, LayerType.ATTENTION]:
                    unsupported_layers.append(node.layer_type.value)

            if unsupported_layers:
                suggestions.append(
                    f"Layers {unsupported_layers} may not be optimized for edge devices"
                )

        # Accuracy vs performance trade-offs
        if (
            quantization in ["int8", "uint8"]
            and performance_metrics.accuracy_estimate < 0.90
        ):
            suggestions.append(
                "Quantization may significantly impact accuracy - consider float16 instead"
            )

        return suggestions

    def compare_configurations(
        self, ir_graph: IRGraph, configurations: List[Dict[str, str]]
    ) -> Dict[str, FastCompileResult]:
        """
        Compare multiple device/quantization configurations.

        Args:
            ir_graph: The IR graph to analyze
            configurations: List of {'device': str, 'quantization': str} configs

        Returns:
            Dictionary mapping config names to FastCompileResult
        """
        results = {}

        for i, config in enumerate(configurations):
            device = config.get("device", "mobile")
            quantization = config.get("quantization", "float32")
            config_name = f"{device}_{quantization}"

            result = self.fast_compile(ir_graph, device, quantization)
            results[config_name] = result

        return results

    def get_optimization_recommendations(self, ir_graph: IRGraph) -> Dict[str, Any]:
        """
        Get comprehensive optimization recommendations for a model.

        Returns:
            Dictionary with optimization recommendations and trade-off analysis
        """
        # Test multiple configurations
        configurations = [
            {"device": "edge", "quantization": "int8"},
            {"device": "edge", "quantization": "float16"},
            {"device": "mobile", "quantization": "float16"},
            {"device": "mobile", "quantization": "float32"},
            {"device": "server", "quantization": "float32"},
        ]

        results = self.compare_configurations(ir_graph, configurations)

        # Analyze trade-offs
        recommendations = {
            "best_for_speed": None,
            "best_for_memory": None,
            "best_for_accuracy": None,
            "best_for_power": None,
            "balanced_recommendation": None,
            "detailed_results": results,
        }

        valid_results = {
            k: v for k, v in results.items() if v.success and v.performance_metrics
        }

        if valid_results:
            # Find best configurations for different criteria
            speed_winner = min(
                valid_results.items(),
                key=lambda x: x[1].performance_metrics.inference_time_ms,
            )
            recommendations["best_for_speed"] = speed_winner[0]

            memory_winner = min(
                valid_results.items(),
                key=lambda x: x[1].performance_metrics.memory_usage_mb,
            )
            recommendations["best_for_memory"] = memory_winner[0]

            accuracy_winner = max(
                valid_results.items(),
                key=lambda x: x[1].performance_metrics.accuracy_estimate,
            )
            recommendations["best_for_accuracy"] = accuracy_winner[0]

            power_winner = min(
                valid_results.items(),
                key=lambda x: x[1].performance_metrics.power_consumption_mw,
            )
            recommendations["best_for_power"] = power_winner[0]

            # Balanced recommendation (weighted score)
            def balanced_score(result):
                metrics = result.performance_metrics
                # Lower is better for time, memory, power; higher is better for accuracy
                score = (
                    -metrics.inference_time_ms * 0.3
                    + -metrics.memory_usage_mb * 0.2
                    + -metrics.power_consumption_mw * 0.2
                    + metrics.accuracy_estimate * 1000 * 0.3  # Scale accuracy
                )
                return score

            balanced_winner = max(
                valid_results.items(), key=lambda x: balanced_score(x[1])
            )
            recommendations["balanced_recommendation"] = balanced_winner[0]

        return recommendations


# ============================================================================
# Utility Functions and CLI Interface
# ============================================================================


def create_sample_ir_graph() -> IRGraph:
    """Create a sample IR graph for testing purposes."""
    from semantic_analyzer.error_types import SourceLocation
    from semantic_analyzer.ir_nodes import (
        ActivationType,
        TensorShape,
        create_conv2d_node,
        create_dense_node,
        create_input_node,
    )

    graph = IRGraph()

    # Input layer
    input_node = create_input_node(
        node_id="input_1",
        shape=TensorShape((224, 224, 3)),
        dtype=DataType.FLOAT32,
        name="input",
        location=SourceLocation(line=1, column=1, file_path="sample.dsl"),
    )
    graph.add_node(input_node)

    # Conv2D layer
    conv_node = create_conv2d_node(
        node_id="conv_1",
        filters=64,
        kernel_size=(3, 3),
        activation=ActivationType.RELU,
        name="conv2d_1",
        location=SourceLocation(line=2, column=1, file_path="sample.dsl"),
    )
    graph.add_node(conv_node)

    # Dense layer
    dense_node = create_dense_node(
        node_id="dense_1",
        units=128,
        activation=ActivationType.RELU,
        name="dense_1",
        location=SourceLocation(line=3, column=1, file_path="sample.dsl"),
    )
    graph.add_node(dense_node)

    return graph


def demo_fast_compiler():
    """Demonstrate the fast compiler capabilities."""
    print("🚀 EdgeFlow Fast Compiler Demo")
    print("=" * 50)

    # Create sample IR graph
    ir_graph = create_sample_ir_graph()

    # Initialize fast compiler
    compiler = EdgeFlowFastCompiler()

    # Test single compilation
    print("\n📊 Single Configuration Analysis:")
    result = compiler.fast_compile(
        ir_graph, target_device="mobile", quantization="float32"
    )

    print(f"✅ Success: {result.success}")
    print(f"⏱️  Compile Time: {result.compile_time_ms:.2f}ms")

    if result.errors:
        print(f"❌ Errors:")
        for error in result.errors:
            print(f"   • {error}")

    if result.warnings:
        print(f"⚠️  Warnings:")
        for warning in result.warnings:
            print(f"   • {warning}")

    if result.performance_metrics:
        metrics = result.performance_metrics
        print(f"📈 Performance Metrics:")
        print(f"   Model Size: {metrics.model_size_mb:.2f} MB")
        print(f"   Inference Time: {metrics.inference_time_ms:.2f} ms")
        print(f"   Memory Usage: {metrics.memory_usage_mb:.2f} MB")
        print(f"   Power Consumption: {metrics.power_consumption_mw:.2f} mW")
        print(f"   Accuracy Estimate: {metrics.accuracy_estimate:.1%}")

    if result.optimization_suggestions:
        print(f"💡 Optimization Suggestions:")
        for suggestion in result.optimization_suggestions:
            print(f"   • {suggestion}")

    print(f"🔧 Device Compatibility:")
    for device, compatible in result.device_compatibility.items():
        status = "✅" if compatible else "❌"
        print(
            f"   {status} {device.capitalize()}: {'Compatible' if compatible else 'Incompatible'}"
        )

    # Test configuration comparison
    print(f"\n🔍 Configuration Comparison:")
    try:
        recommendations = compiler.get_optimization_recommendations(ir_graph)

        print(f"🏆 Best Configurations:")
        print(f"   Speed: {recommendations.get('best_for_speed', 'N/A')}")
        print(f"   Memory: {recommendations.get('best_for_memory', 'N/A')}")
        print(f"   Accuracy: {recommendations.get('best_for_accuracy', 'N/A')}")
        print(f"   Power: {recommendations.get('best_for_power', 'N/A')}")
        print(f"   Balanced: {recommendations.get('balanced_recommendation', 'N/A')}")

        # Show some detailed results if available
        detailed = recommendations.get("detailed_results", {})
        if detailed:
            print(f"\n📊 Detailed Configuration Results:")
            for config_name, config_result in list(detailed.items())[
                :3
            ]:  # Show first 3
                if config_result.success and config_result.performance_metrics:
                    metrics = config_result.performance_metrics
                    print(f"   {config_name}:")
                    print(
                        f"     Size: {metrics.model_size_mb:.2f}MB, "
                        f"Time: {metrics.inference_time_ms:.2f}ms, "
                        f"Accuracy: {metrics.accuracy_estimate:.1%}"
                    )
                else:
                    print(f"   {config_name}: ❌ Failed compilation")

    except Exception as e:
        print(f"Configuration comparison failed: {e}")

    print(f"\n🎯 Summary:")
    print(f"   • Fast compilation completed in {result.compile_time_ms:.2f}ms")
    print(f"   • Dynamic performance estimation based on device profiles")
    print(f"   • Integrated with existing semantic analyzer")
    print(f"   • Device compatibility checking across edge/mobile/server")
    print(f"   • Intelligent optimization suggestions generated")


def fast_compile_config(config: Dict[str, Any]) -> FastCompileResult:
    """
    Fast compile a configuration for immediate feedback.

    Args:
        config: EdgeFlow configuration dictionary

    Returns:
        FastCompileResult with validation and performance estimates
    """
    try:
        from edgeflow_ir import IRBuilder

        # Build IR graph from config
        ir_builder = IRBuilder()
        ir_graph = ir_builder.build_from_config(config)

        # Get target device and quantization from config
        target_device = config.get("target_device", "mobile")
        quantization = config.get("quantize", "float32")

        # Initialize fast compiler and run compilation
        compiler = EdgeFlowFastCompiler()
        result = compiler.fast_compile(ir_graph, target_device, quantization)

        return result

    except Exception as e:
        logger.error(f"Fast compile config failed: {e}")
        return FastCompileResult(
            success=False,
            errors=[f"Configuration compilation failed: {str(e)}"],
            warnings=[],
            performance_metrics=None,
            compile_time_ms=0.0,
            device_compatibility={},
            optimization_suggestions=[],
        )


if __name__ == "__main__":
    demo_fast_compiler()
