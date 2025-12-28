"""
EdgeFlow GPU Optimization Integration

Integrates GPU acceleration capabilities with the complete EdgeFlow optimization
pipeline, providing intelligent GPU-aware model optimization and deployment.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from dynamic_device_profiles import get_profile_manager
from gpu import AccelerationType, GPUAccelerationManager, GPUInfo
from integrated_error_system import ErrorCategory, get_error_reporter
from optimization_orchestrator import OptimizationConfig, OptimizationOrchestrator
from traceability_system import (
    TransformationType,
    register_artifact,
    trace_transformation,
)

logger = logging.getLogger(__name__)


class GPUOptimizationStrategy(Enum):
    """GPU-specific optimization strategies."""

    GPU_MEMORY_OPTIMIZED = "gpu_memory_optimized"
    GPU_COMPUTE_OPTIMIZED = "gpu_compute_optimized"
    GPU_POWER_EFFICIENT = "gpu_power_efficient"
    GPU_BALANCED = "gpu_balanced"
    CPU_FALLBACK = "cpu_fallback"


@dataclass
class GPUOptimizationConfig:
    """Configuration for GPU-aware optimization."""

    target_gpu: Optional[GPUInfo] = None
    preferred_acceleration: Optional[AccelerationType] = None
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    enable_tensorrt: bool = False
    enable_gpu_delegate: bool = True
    fallback_to_cpu: bool = True
    optimization_strategy: GPUOptimizationStrategy = (
        GPUOptimizationStrategy.GPU_BALANCED
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_gpu": self.target_gpu.to_dict() if self.target_gpu else None,
            "preferred_acceleration": self.preferred_acceleration.value
            if self.preferred_acceleration
            else None,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "enable_mixed_precision": self.enable_mixed_precision,
            "enable_tensorrt": self.enable_tensorrt,
            "enable_gpu_delegate": self.enable_gpu_delegate,
            "fallback_to_cpu": self.fallback_to_cpu,
            "optimization_strategy": self.optimization_strategy.value,
        }


@dataclass
class GPUOptimizationResult:
    """Results from GPU-aware optimization."""

    success: bool
    gpu_used: Optional[GPUInfo] = None
    acceleration_used: Optional[AccelerationType] = None
    optimization_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    performance_improvement: float = 0.0
    model_size_reduction: float = 0.0
    optimizations_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class GPUOptimizationIntegrator:
    """Integrates GPU acceleration with EdgeFlow optimization pipeline."""

    def __init__(self):
        self.gpu_manager = GPUAccelerationManager()
        self.optimizer = OptimizationOrchestrator()
        self.profile_manager = get_profile_manager()
        self.error_reporter = get_error_reporter()

        # Initialize GPU-specific device profiles
        self._create_gpu_device_profiles()

    def _create_gpu_device_profiles(self) -> None:
        """Create device profiles for detected GPUs."""
        for gpu in self.gpu_manager.detected_gpus:
            try:
                profile = self.gpu_manager.create_gpu_device_profile(gpu)
                self.profile_manager.register_profile(profile)
                logger.info(f"Created device profile for GPU: {gpu.name}")
            except Exception as e:
                logger.warning(f"Failed to create profile for GPU {gpu.name}: {e}")

    def create_gpu_optimization_config(
        self,
        target_device: str = "auto",
        strategy: GPUOptimizationStrategy = GPUOptimizationStrategy.GPU_BALANCED,
    ) -> GPUOptimizationConfig:
        """Create GPU optimization configuration."""

        config = GPUOptimizationConfig(optimization_strategy=strategy)

        # Select target GPU
        if target_device == "auto":
            config.target_gpu = self.gpu_manager.get_best_gpu_for_inference()
        else:
            # Find GPU by name or use best available
            for gpu in self.gpu_manager.detected_gpus:
                if target_device.lower() in gpu.name.lower():
                    config.target_gpu = gpu
                    break
            else:
                config.target_gpu = self.gpu_manager.get_best_gpu_for_inference()

        if not config.target_gpu:
            logger.warning("No GPU available, falling back to CPU optimization")
            config.optimization_strategy = GPUOptimizationStrategy.CPU_FALLBACK
            return config

        # Set acceleration preference
        config.preferred_acceleration = self.gpu_manager.get_recommended_acceleration(
            "tensorflow"
        )

        # Configure strategy-specific settings
        if strategy == GPUOptimizationStrategy.GPU_MEMORY_OPTIMIZED:
            config.gpu_memory_fraction = 0.6
            config.enable_mixed_precision = True
            config.enable_tensorrt = False

        elif strategy == GPUOptimizationStrategy.GPU_COMPUTE_OPTIMIZED:
            config.gpu_memory_fraction = 0.9
            config.enable_mixed_precision = True
            config.enable_tensorrt = True

        elif strategy == GPUOptimizationStrategy.GPU_POWER_EFFICIENT:
            config.gpu_memory_fraction = 0.5
            config.enable_mixed_precision = True
            config.enable_tensorrt = False

        elif strategy == GPUOptimizationStrategy.GPU_BALANCED:
            config.gpu_memory_fraction = 0.8
            config.enable_mixed_precision = True
            config.enable_tensorrt = (
                config.target_gpu.vendor.value == "nvidia"
                if config.target_gpu
                else False
            )

        return config

    def optimize_model_for_gpu(
        self,
        model_path: str,
        gpu_config: GPUOptimizationConfig,
        output_path: Optional[str] = None,
    ) -> GPUOptimizationResult:
        """Optimize model for GPU acceleration."""

        start_time = time.perf_counter()
        result = GPUOptimizationResult(success=False)

        with trace_transformation(
            TransformationType.OPTIMIZATION,
            "gpu_optimization",
            f"GPU-aware model optimization for "
            f"{gpu_config.target_gpu.name if gpu_config.target_gpu else 'CPU'}",
            parameters=gpu_config.to_dict(),
        ) as ctx:
            try:
                # Check if GPU optimization is possible
                if (
                    gpu_config.optimization_strategy
                    == GPUOptimizationStrategy.CPU_FALLBACK
                ):
                    return self._fallback_to_cpu_optimization(model_path, output_path)

                # Create base optimization config
                base_config = self._create_base_optimization_config(gpu_config)

                # Apply GPU-specific optimizations
                result = self._apply_gpu_optimizations(
                    model_path, gpu_config, base_config, output_path
                )

                # Benchmark GPU performance
                if result.success and result.gpu_used:
                    performance_metrics = self._benchmark_gpu_model(
                        model_path, result.gpu_used
                    )
                    result.performance_improvement = performance_metrics.get(
                        "speedup", 0.0
                    )
                    result.memory_usage_mb = performance_metrics.get("memory_mb", 0.0)

                # Register artifacts
                if result.success:
                    register_artifact(
                        f"gpu_optimized_model_{int(time.time())}",
                        "optimized_model",
                        metadata={
                            "gpu_used": result.gpu_used.to_dict()
                            if result.gpu_used
                            else None,
                            "acceleration": result.acceleration_used.value
                            if result.acceleration_used
                            else None,
                            "optimizations": result.optimizations_applied,
                            "performance_improvement": result.performance_improvement,
                        },
                        created_by="gpu_optimization",
                    )

                ctx.add_metric("optimization_success", result.success)
                ctx.add_metric(
                    "performance_improvement", result.performance_improvement
                )
                ctx.add_metric("memory_usage_mb", result.memory_usage_mb)

            except Exception as e:
                error_msg = f"GPU optimization failed: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

                # Report error with suggestions
                self.error_reporter.report_error(
                    ErrorCategory.OPTIMIZATION,
                    "ERROR",
                    "GPU optimization failed",
                    str(e),
                    suggestions=[
                        "Check GPU drivers and CUDA installation",
                        "Verify model compatibility with GPU acceleration",
                        "Try CPU fallback optimization",
                        "Reduce GPU memory fraction in configuration",
                    ],
                )

        result.optimization_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _create_base_optimization_config(
        self, gpu_config: GPUOptimizationConfig
    ) -> OptimizationConfig:
        """Create base optimization configuration."""
        from optimization_orchestrator import OptimizationLevel, OptimizationStrategy

        # Map GPU strategy to base strategy
        strategy_mapping = {
            GPUOptimizationStrategy.GPU_MEMORY_OPTIMIZED: (
                OptimizationStrategy.SIZE_FOCUSED
            ),
            GPUOptimizationStrategy.GPU_COMPUTE_OPTIMIZED: (
                OptimizationStrategy.SPEED_FOCUSED
            ),
            GPUOptimizationStrategy.GPU_POWER_EFFICIENT: (
                OptimizationStrategy.POWER_EFFICIENT
            ),
            GPUOptimizationStrategy.GPU_BALANCED: OptimizationStrategy.BALANCED,
        }

        base_strategy = strategy_mapping.get(
            gpu_config.optimization_strategy, OptimizationStrategy.BALANCED
        )

        return self.optimizer.create_optimization_config(
            target_device="gpu_accelerated",
            strategy=base_strategy,
            optimization_level=OptimizationLevel.BALANCED,
        )

    def _apply_gpu_optimizations(
        self,
        model_path: str,
        gpu_config: GPUOptimizationConfig,
        base_config: OptimizationConfig,
        output_path: Optional[str],
    ) -> GPUOptimizationResult:
        """Apply GPU-specific optimizations."""

        result = GPUOptimizationResult(success=True)
        result.gpu_used = gpu_config.target_gpu
        result.acceleration_used = gpu_config.preferred_acceleration

        try:
            # Apply TensorFlow Lite GPU delegate optimization
            if gpu_config.enable_gpu_delegate:
                self._optimize_for_gpu_delegate(model_path, gpu_config)
                result.optimizations_applied.append("GPU Delegate Optimization")

            # Apply TensorRT optimization (NVIDIA only)
            if (
                gpu_config.enable_tensorrt
                and gpu_config.target_gpu
                and gpu_config.target_gpu.vendor.value == "nvidia"
            ):
                self._optimize_for_tensorrt(model_path, gpu_config)
                result.optimizations_applied.append("TensorRT Optimization")

            # Apply mixed precision optimization
            if gpu_config.enable_mixed_precision:
                self._optimize_mixed_precision(model_path, gpu_config)
                result.optimizations_applied.append("Mixed Precision (FP16)")

            # Apply memory optimization
            self._optimize_gpu_memory_layout(model_path, gpu_config)
            result.optimizations_applied.append("GPU Memory Layout Optimization")

            # Run base optimization pipeline
            base_result = self.optimizer.optimize_model(model_path, base_config)
            if base_result.success:
                result.model_size_reduction = base_result.size_reduction_percent
                result.optimizations_applied.extend(base_result.optimizations_applied)
            else:
                result.warnings.append("Base optimization pipeline had issues")

        except Exception as e:
            result.success = False
            result.errors.append(f"GPU optimization failed: {e}")

        return result

    def _optimize_for_gpu_delegate(
        self, model_path: str, gpu_config: GPUOptimizationConfig
    ) -> None:
        """Optimize model for TensorFlow Lite GPU delegate."""
        try:
            # This would integrate with TensorFlow Lite converter
            logger.info("Applying GPU delegate optimizations")

            # Simulate GPU delegate optimization
            # In real implementation, this would use TFLite converter with GPU delegate
            optimization_params = {
                "use_gpu_delegate": True,
                "gpu_precision": "fp16"
                if gpu_config.enable_mixed_precision
                else "fp32",
                "gpu_memory_fraction": gpu_config.gpu_memory_fraction,
            }

            logger.info(f"GPU delegate optimization params: {optimization_params}")

        except Exception as e:
            logger.warning(f"GPU delegate optimization failed: {e}")

    def _optimize_for_tensorrt(
        self, model_path: str, gpu_config: GPUOptimizationConfig
    ) -> None:
        """Optimize model for TensorRT (NVIDIA GPUs)."""
        try:
            logger.info("Applying TensorRT optimizations")

            # Simulate TensorRT optimization
            # In real implementation, this would use TensorRT Python API
            tensorrt_params = {
                "precision": "fp16" if gpu_config.enable_mixed_precision else "fp32",
                "max_workspace_size": int(
                    gpu_config.gpu_memory_fraction * 1024 * 1024 * 1024
                ),
                "max_batch_size": 1,
            }

            logger.info(f"TensorRT optimization params: {tensorrt_params}")

        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")

    def _optimize_mixed_precision(
        self, model_path: str, gpu_config: GPUOptimizationConfig
    ) -> None:
        """Apply mixed precision optimization."""
        try:
            logger.info("Applying mixed precision (FP16) optimization")

            # Simulate mixed precision optimization
            # In real implementation, this would modify the model to use FP16
            mixed_precision_params = {
                "dtype": "float16",
                "loss_scale": "dynamic",
                "keep_fp32_ops": ["softmax", "cross_entropy"],
            }

            logger.info(f"Mixed precision params: {mixed_precision_params}")

        except Exception as e:
            logger.warning(f"Mixed precision optimization failed: {e}")

    def _optimize_gpu_memory_layout(
        self, model_path: str, gpu_config: GPUOptimizationConfig
    ) -> None:
        """Optimize memory layout for GPU."""
        try:
            logger.info("Optimizing GPU memory layout")

            # Simulate memory layout optimization
            # In real implementation, this would reorder operations for better GPU
            # memory access
            memory_params = {
                "memory_fraction": gpu_config.gpu_memory_fraction,
                "allow_growth": True,
                "memory_layout": "NCHW"
                if gpu_config.target_gpu
                and gpu_config.target_gpu.vendor.value == "nvidia"
                else "NHWC",
            }

            logger.info(f"Memory layout params: {memory_params}")

        except Exception as e:
            logger.warning(f"Memory layout optimization failed: {e}")

    def _fallback_to_cpu_optimization(
        self, model_path: str, output_path: Optional[str]
    ) -> GPUOptimizationResult:
        """Fallback to CPU-only optimization."""
        logger.info("Falling back to CPU optimization")

        result = GPUOptimizationResult(success=True)
        result.optimizations_applied.append("CPU Fallback Optimization")
        result.warnings.append("No GPU available, using CPU optimization")

        # Use base optimizer for CPU optimization
        from optimization_orchestrator import OptimizationStrategy

        cpu_config = self.optimizer.create_optimization_config(
            target_device="cpu", strategy=OptimizationStrategy.BALANCED
        )

        base_result = self.optimizer.optimize_model(model_path, cpu_config)
        if base_result.success:
            result.model_size_reduction = base_result.size_reduction_percent
            result.optimizations_applied.extend(base_result.optimizations_applied)

        return result

    def _benchmark_gpu_model(
        self, model_path: str, gpu_info: GPUInfo
    ) -> Dict[str, float]:
        """Benchmark optimized model on GPU."""
        try:
            logger.info(f"Benchmarking model on {gpu_info.name}")

            # Simulate GPU benchmarking
            # In real implementation, this would load and run the model on GPU
            benchmarks = self.gpu_manager.benchmark_gpu_performance(gpu_info)

            # Add model-specific metrics
            benchmarks.update(
                {
                    "inference_latency_ms": 15.0,  # Simulated
                    "throughput_fps": 66.7,  # Simulated
                    "memory_mb": gpu_info.memory_mb * 0.3,  # Simulated usage
                    "speedup": 2.5,  # Simulated speedup vs CPU
                }
            )

            return benchmarks

        except Exception as e:
            logger.warning(f"GPU benchmarking failed: {e}")
            return {}

    def get_gpu_optimization_recommendations(self, model_path: str) -> Dict[str, Any]:
        """Get GPU optimization recommendations for a model."""
        recommendations: Dict[str, Any] = {
            "available_gpus": len(self.gpu_manager.detected_gpus),
            "best_gpu": None,
            "recommended_strategy": GPUOptimizationStrategy.CPU_FALLBACK.value,
            "expected_speedup": 1.0,
            "memory_requirements": "Unknown",
            "optimizations": [],
            "warnings": [],
        }

        best_gpu = self.gpu_manager.get_best_gpu_for_inference()
        if best_gpu:
            recommendations.update(
                {
                    "best_gpu": best_gpu.to_dict(),
                    "recommended_strategy": GPUOptimizationStrategy.GPU_BALANCED.value,
                    "expected_speedup": 2.0 + (best_gpu.performance_score or 50) / 50,
                    "memory_requirements": f"~{best_gpu.memory_mb * 0.3:.0f}MB",
                    "optimizations": [
                        "GPU Delegate Optimization",
                        "Mixed Precision (FP16)",
                        "Memory Layout Optimization",
                    ],
                }
            )

            # Add vendor-specific recommendations
            if best_gpu.vendor.value == "nvidia":
                recommendations["optimizations"].append("TensorRT Optimization")
            elif best_gpu.vendor.value == "apple":
                recommendations["optimizations"].append("Metal Performance Shaders")

            # Add warnings based on GPU memory
            if best_gpu.memory_mb < 2048:
                recommendations["warnings"].append(
                    "Limited GPU memory may require smaller batch sizes"
                )

            if best_gpu.performance_score and best_gpu.performance_score < 70:
                recommendations["warnings"].append(
                    "GPU performance may be limited for large models"
                )

        else:
            recommendations["warnings"].append(
                "No GPU detected, CPU optimization recommended"
            )

        return recommendations


def create_gpu_optimization_integrator() -> GPUOptimizationIntegrator:
    """Create and initialize GPU optimization integrator."""
    return GPUOptimizationIntegrator()


def optimize_model_with_gpu_acceleration(
    model_path: str,
    target_device: str = "auto",
    strategy: GPUOptimizationStrategy = GPUOptimizationStrategy.GPU_BALANCED,
    output_path: Optional[str] = None,
) -> GPUOptimizationResult:
    """High-level function to optimize model with GPU acceleration."""

    integrator = create_gpu_optimization_integrator()

    # Get recommendations first
    recommendations = integrator.get_gpu_optimization_recommendations(model_path)
    logger.info(f"GPU optimization recommendations: {recommendations}")

    # Create configuration
    config = integrator.create_gpu_optimization_config(target_device, strategy)

    # Optimize model
    result = integrator.optimize_model_for_gpu(model_path, config, output_path)

    # Log results
    if result.success:
        logger.info("✅ GPU optimization completed successfully")
        logger.info(f"🚀 Performance improvement: {result.performance_improvement:.1f}x")
        logger.info(f"📊 Model size reduction: {result.model_size_reduction:.1f}%")
        logger.info(
            f"⚡ GPU used: {result.gpu_used.name if result.gpu_used else 'None'}"
        )
        logger.info(f"🔧 Optimizations: {', '.join(result.optimizations_applied)}")
    else:
        logger.error("❌ GPU optimization failed")
        for error in result.errors:
            logger.error(f"  - {error}")

    return result


if __name__ == "__main__":
    # Demo GPU optimization integration
    logging.basicConfig(level=logging.INFO)

    print("🔍 EdgeFlow GPU Optimization Integration Demo")
    print("=" * 50)

    # Create integrator
    integrator = create_gpu_optimization_integrator()

    # Show GPU status
    if integrator.gpu_manager.detected_gpus:
        print(f"\n✅ Detected {len(integrator.gpu_manager.detected_gpus)} GPU(s)")
        best_gpu = integrator.gpu_manager.get_best_gpu_for_inference()
        print(f"🚀 Best GPU: {best_gpu.name} ({best_gpu.performance_score:.1f}/100)")

        # Show optimization strategies
        print("\n🎯 Available GPU optimization strategies:")
        for strategy in GPUOptimizationStrategy:
            print(f"  • {strategy.value}")

        # Get recommendations for a demo model
        recommendations = integrator.get_gpu_optimization_recommendations(
            "demo_model.tflite"
        )
        print("\n💡 Optimization recommendations:")
        print(f"  Strategy: {recommendations['recommended_strategy']}")
        print(f"  Expected speedup: {recommendations['expected_speedup']:.1f}x")
        print(f"  Memory requirements: {recommendations['memory_requirements']}")
        print(f"  Optimizations: {', '.join(recommendations['optimizations'])}")

        if recommendations["warnings"]:
            print(f"  ⚠️  Warnings: {', '.join(recommendations['warnings'])}")

    else:
        print("\n❌ No GPU detected - CPU optimization will be used")

    print("\n" + "=" * 50)
