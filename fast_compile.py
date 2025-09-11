"""EdgeFlow Fast Compile System

This module implements fast compile-feedback cycles for rapid developer iteration,
providing immediate feedback on configuration changes without full optimization.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from validator import EdgeFlowValidator

logger = logging.getLogger(__name__)


class FastCompileResult:
    """Result from a fast compilation cycle."""
    
    def __init__(self, success: bool, errors: List[str], warnings: List[str], 
                 estimated_impact: Dict[str, Any], compile_time_ms: float):
        self.success = success
        self.errors = errors
        self.warnings = warnings
        self.estimated_impact = estimated_impact
        self.compile_time_ms = compile_time_ms


class EdgeFlowFastCompiler:
    """Provides fast compilation feedback for rapid development iteration."""
    
    def __init__(self):
        self.validator = EdgeFlowValidator()
        
        # Performance estimation models (simplified)
        self.quantization_impact = {
            "int8": {"size_reduction": 0.75, "speed_improvement": 2.5, "memory_reduction": 0.6},
            "float16": {"size_reduction": 0.5, "speed_improvement": 1.3, "memory_reduction": 0.4},
            "none": {"size_reduction": 0.0, "speed_improvement": 1.0, "memory_reduction": 0.0}
        }
        
        self.device_characteristics = {
            "raspberry_pi": {"base_speed": 1.0, "memory_efficiency": 0.8, "power_efficiency": 0.9},
            "jetson_nano": {"base_speed": 2.0, "memory_efficiency": 1.2, "power_efficiency": 0.7},
            "jetson_xavier": {"base_speed": 4.0, "memory_efficiency": 1.5, "power_efficiency": 0.6},
            "cortex_m4": {"base_speed": 0.5, "memory_efficiency": 0.6, "power_efficiency": 0.95},
            "cortex_m7": {"base_speed": 0.8, "memory_efficiency": 0.7, "power_efficiency": 0.9},
            "cpu": {"base_speed": 1.5, "memory_efficiency": 1.0, "power_efficiency": 0.8},
            "gpu": {"base_speed": 3.0, "memory_efficiency": 1.3, "power_efficiency": 0.5}
        }
    
    def fast_compile(self, config: Dict[str, Any]) -> FastCompileResult:
        """Perform fast compilation with immediate feedback.
        
        Args:
            config: EdgeFlow configuration dictionary
            
        Returns:
            FastCompileResult with validation and impact estimation
        """
        start_time = time.time()
        
        # Step 1: Fast validation (no model loading)
        is_valid, errors = self.validator.early_validation(config)
        warnings = []
        
        if not is_valid:
            compile_time = (time.time() - start_time) * 1000
            return FastCompileResult(False, errors, warnings, {}, compile_time)
        
        # Step 2: Estimate optimization impact
        estimated_impact = self._estimate_optimization_impact(config)
        
        # Step 3: Generate warnings for potential issues
        warnings = self._generate_warnings(config, estimated_impact)
        
        compile_time = (time.time() - start_time) * 1000
        
        return FastCompileResult(True, [], warnings, estimated_impact, compile_time)
    
    def _estimate_optimization_impact(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the impact of optimizations without running them."""
        quantize = config.get("quantize", "none")
        device = config.get("target_device", "cpu")
        optimize_for = config.get("optimize_for", "balanced")
        enable_pruning = config.get("enable_pruning", False)
        enable_fusion = config.get("enable_fusion", True)
        
        # Base impact from quantization
        quant_impact = self.quantization_impact.get(quantize, self.quantization_impact["none"])
        
        # Device-specific adjustments
        device_chars = self.device_characteristics.get(device, self.device_characteristics["cpu"])
        
        # Calculate estimated improvements
        size_reduction = quant_impact["size_reduction"]
        speed_improvement = quant_impact["speed_improvement"] * device_chars["base_speed"]
        memory_reduction = quant_impact["memory_reduction"] * device_chars["memory_efficiency"]
        
        # Adjust for pruning
        if enable_pruning:
            pruning_sparsity = config.get("pruning_sparsity", 0.5)
            size_reduction += pruning_sparsity * 0.3  # Up to 30% additional from pruning
            speed_improvement *= (1 + pruning_sparsity * 0.2)  # Up to 20% speed improvement
        
        # Adjust for fusion
        if enable_fusion:
            speed_improvement *= 1.2  # 20% improvement from fusion
            memory_reduction += 0.1  # 10% additional memory reduction
        
        # Optimize-for specific adjustments
        if optimize_for == "latency":
            speed_improvement *= 1.3  # Prioritize speed
        elif optimize_for == "memory":
            memory_reduction += 0.2  # Prioritize memory
            size_reduction += 0.1
        elif optimize_for == "size":
            size_reduction += 0.2  # Prioritize size
        
        # Cap improvements at reasonable limits
        size_reduction = min(size_reduction, 0.9)  # Max 90% reduction
        speed_improvement = min(speed_improvement, 10.0)  # Max 10x speedup
        memory_reduction = min(memory_reduction, 0.8)  # Max 80% memory reduction
        
        return {
            "estimated_size_reduction_percent": size_reduction * 100,
            "estimated_speed_improvement_factor": speed_improvement,
            "estimated_memory_reduction_percent": memory_reduction * 100,
            "estimated_power_efficiency": device_chars["power_efficiency"],
            "optimization_confidence": self._calculate_confidence(config, quant_impact, device_chars)
        }
    
    def _calculate_confidence(self, config: Dict[str, Any], quant_impact: Dict[str, float], 
                            device_chars: Dict[str, float]) -> float:
        """Calculate confidence in the optimization estimates."""
        confidence = 0.8  # Base confidence
        
        # Increase confidence for well-supported combinations
        quantize = config.get("quantize", "none")
        device = config.get("target_device", "cpu")
        
        # High confidence combinations
        if device == "raspberry_pi" and quantize == "int8":
            confidence += 0.15
        elif device in ["jetson_nano", "jetson_xavier"] and quantize in ["int8", "float16"]:
            confidence += 0.1
        elif device in ["cortex_m4", "cortex_m7"] and quantize == "int8":
            confidence += 0.1
        
        # Decrease confidence for edge cases
        if device == "cortex_m4" and quantize == "float16":
            confidence -= 0.3  # No FP16 support
        
        # Adjust for model format
        model_path = config.get("model", "")
        if model_path.endswith((".h5", ".keras")) and quantize == "int8":
            confidence += 0.05  # Good quantization support
        
        return max(0.3, min(1.0, confidence))  # Clamp between 0.3 and 1.0
    
    def _generate_warnings(self, config: Dict[str, Any], estimated_impact: Dict[str, Any]) -> List[str]:
        """Generate warnings based on configuration and estimated impact."""
        warnings = []
        
        device = config.get("target_device", "cpu")
        quantize = config.get("quantize", "none")
        memory_limit = config.get("memory_limit")
        model_path = config.get("model", "")
        
        # Device-specific warnings
        if device == "cortex_m4" and quantize == "float16":
            warnings.append("FLOAT16 quantization not supported on Cortex-M4 (no FP16 unit)")
        
        if device in ["cortex_m4", "cortex_m7"] and memory_limit and memory_limit > 512:
            warnings.append(f"Memory limit ({memory_limit}MB) may be too high for {device}")
        
        # Model format warnings
        if model_path.endswith((".pth", ".pt")) and quantize == "int8":
            warnings.append("PyTorch models may need conversion to ONNX for optimal INT8 quantization")
        
        # Performance warnings
        confidence = estimated_impact.get("optimization_confidence", 0.8)
        if confidence < 0.6:
            warnings.append("Low confidence in optimization estimates - results may vary significantly")
        
        # Memory constraint warnings
        if memory_limit:
            estimated_size_reduction = estimated_impact.get("estimated_size_reduction_percent", 0)
            if estimated_size_reduction < 50 and memory_limit < 256:
                warnings.append("Consider more aggressive optimization for tight memory constraints")
        
        return warnings
    
    def compare_configurations(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two configurations and estimate relative performance."""
        result1 = self.fast_compile(config1)
        result2 = self.fast_compile(config2)
        
        if not result1.success or not result2.success:
            return {"error": "One or both configurations are invalid"}
        
        impact1 = result1.estimated_impact
        impact2 = result2.estimated_impact
        
        return {
            "config1": {
                "size_reduction": impact1.get("estimated_size_reduction_percent", 0),
                "speed_improvement": impact1.get("estimated_speed_improvement_factor", 1.0),
                "memory_reduction": impact1.get("estimated_memory_reduction_percent", 0),
                "confidence": impact1.get("optimization_confidence", 0.8)
            },
            "config2": {
                "size_reduction": impact2.get("estimated_size_reduction_percent", 0),
                "speed_improvement": impact2.get("estimated_speed_improvement_factor", 1.0),
                "memory_reduction": impact2.get("estimated_memory_reduction_percent", 0),
                "confidence": impact2.get("optimization_confidence", 0.8)
            },
            "comparison": {
                "size_improvement": impact2.get("estimated_size_reduction_percent", 0) - impact1.get("estimated_size_reduction_percent", 0),
                "speed_improvement": impact2.get("estimated_speed_improvement_factor", 1.0) - impact1.get("estimated_speed_improvement_factor", 1.0),
                "memory_improvement": impact2.get("estimated_memory_reduction_percent", 0) - impact1.get("estimated_memory_reduction_percent", 0)
            },
            "recommendation": self._get_configuration_recommendation(impact1, impact2)
        }
    
    def _get_configuration_recommendation(self, impact1: Dict[str, Any], impact2: Dict[str, Any]) -> str:
        """Get a recommendation between two configurations."""
        size1 = impact1.get("estimated_size_reduction_percent", 0)
        size2 = impact2.get("estimated_size_reduction_percent", 0)
        speed1 = impact1.get("estimated_speed_improvement_factor", 1.0)
        speed2 = impact2.get("estimated_speed_improvement_factor", 1.0)
        conf1 = impact1.get("optimization_confidence", 0.8)
        conf2 = impact2.get("optimization_confidence", 0.8)
        
        # Simple scoring system
        score1 = size1 + (speed1 - 1) * 50 + conf1 * 100
        score2 = size2 + (speed2 - 1) * 50 + conf2 * 100
        
        if score2 > score1 * 1.1:  # 10% improvement threshold
            return "Config 2 is significantly better"
        elif score1 > score2 * 1.1:
            return "Config 1 is significantly better"
        else:
            return "Configurations are similar - choose based on specific requirements"


def fast_compile_config(config: Dict[str, Any]) -> FastCompileResult:
    """Fast compile a configuration with immediate feedback.
    
    Args:
        config: EdgeFlow configuration dictionary
        
    Returns:
        FastCompileResult with validation and impact estimation
    """
    compiler = EdgeFlowFastCompiler()
    return compiler.fast_compile(config)


def compare_configurations(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two configurations for relative performance.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Comparison results with recommendations
    """
    compiler = EdgeFlowFastCompiler()
    return compiler.compare_configurations(config1, config2)


if __name__ == "__main__":
    # Test the fast compiler
    test_config = {
        "model": "test_model.tflite",
        "quantize": "int8",
        "target_device": "raspberry_pi",
        "optimize_for": "latency",
        "enable_pruning": True,
        "enable_fusion": True,
        "memory_limit": 64
    }
    
    result = fast_compile_config(test_config)
    
    print(f"Fast Compile Result:")
    print(f"Success: {result.success}")
    print(f"Compile Time: {result.compile_time_ms:.2f}ms")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print(f"Estimated Impact: {result.estimated_impact}")
