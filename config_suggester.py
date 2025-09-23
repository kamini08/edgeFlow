"""EdgeFlow Configuration Suggester

This module provides intelligent configuration suggestions and fallbacks for EdgeFlow DSL,
helping users optimize their configurations for specific devices and use cases.

Key Features:
- Intelligent parameter suggestions based on device capabilities
- Automatic fallback configurations for common scenarios
- Performance-optimized defaults for different devices
- Configuration templates for common use cases
- Smart parameter tuning recommendations
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from static_validator import EdgeFlowStaticValidator, ValidationResult

logger = logging.getLogger(__name__)


class UseCase(Enum):
    """Common use cases for EdgeFlow configurations."""

    REAL_TIME_INFERENCE = "real_time_inference"
    BATCH_PROCESSING = "batch_processing"
    MOBILE_DEPLOYMENT = "mobile_deployment"
    IOT_SENSOR = "iot_sensor"
    EDGE_SERVER = "edge_server"
    EMBEDDED_SYSTEM = "embedded_system"
    RESEARCH_PROTOTYPE = "research_prototype"
    PRODUCTION_DEPLOYMENT = "production_deployment"


class PerformanceProfile(Enum):
    """Performance profiles for different optimization strategies."""

    MAXIMUM_SPEED = "maximum_speed"
    MINIMUM_MEMORY = "minimum_memory"
    MAXIMUM_ACCURACY = "maximum_accuracy"
    BALANCED = "balanced"
    ULTRA_LOW_POWER = "ultra_low_power"


@dataclass
class ConfigurationSuggestion:
    """Represents a configuration suggestion with reasoning."""

    parameter: str
    current_value: Any
    suggested_value: Any
    confidence: float  # 0.0 to 1.0
    reasoning: str
    impact: str
    alternatives: List[Tuple[Any, str]] = None  # (value, description)


@dataclass
class OptimizationRecommendation:
    """Represents an optimization recommendation."""

    category: str
    title: str
    description: str
    suggested_changes: List[ConfigurationSuggestion]
    expected_improvement: Dict[str, float]  # metric -> improvement percentage
    implementation_effort: str  # "low", "medium", "high"


class EdgeFlowConfigSuggester:
    """Intelligent configuration suggester for EdgeFlow."""

    def __init__(self):
        self.validator = EdgeFlowStaticValidator()
        self.use_case_templates = self._initialize_use_case_templates()
        self.performance_profiles = self._initialize_performance_profiles()
        self.device_optimizations = self._initialize_device_optimizations()

    def _initialize_use_case_templates(self) -> Dict[UseCase, Dict[str, Any]]:
        """Initialize configuration templates for common use cases."""
        return {
            UseCase.REAL_TIME_INFERENCE: {
                "optimize_for": "latency",
                "buffer_size": 1,
                "enable_fusion": True,
                "quantize": "int8",
                "enable_pruning": False,
                "memory_limit": 256,
                "reasoning": "Optimized for minimal latency with single-frame processing",
            },
            UseCase.BATCH_PROCESSING: {
                "optimize_for": "memory",
                "buffer_size": 64,
                "enable_fusion": True,
                "quantize": "float16",
                "enable_pruning": True,
                "pruning_sparsity": 0.3,
                "memory_limit": 1024,
                "reasoning": "Optimized for processing multiple frames efficiently",
            },
            UseCase.MOBILE_DEPLOYMENT: {
                "optimize_for": "balanced",
                "buffer_size": 8,
                "enable_fusion": True,
                "quantize": "int8",
                "enable_pruning": True,
                "pruning_sparsity": 0.2,
                "memory_limit": 128,
                "reasoning": "Balanced optimization for mobile devices with limited resources",
            },
            UseCase.IOT_SENSOR: {
                "optimize_for": "memory",
                "buffer_size": 1,
                "enable_fusion": False,
                "quantize": "int8",
                "enable_pruning": True,
                "pruning_sparsity": 0.5,
                "memory_limit": 64,
                "reasoning": "Ultra-low memory usage for IoT sensors",
            },
            UseCase.EDGE_SERVER: {
                "optimize_for": "balanced",
                "buffer_size": 128,
                "enable_fusion": True,
                "quantize": "float16",
                "enable_pruning": True,
                "pruning_sparsity": 0.4,
                "memory_limit": 2048,
                "reasoning": "High-performance edge server configuration",
            },
            UseCase.EMBEDDED_SYSTEM: {
                "optimize_for": "memory",
                "buffer_size": 4,
                "enable_fusion": False,
                "quantize": "int8",
                "enable_pruning": True,
                "pruning_sparsity": 0.6,
                "memory_limit": 32,
                "reasoning": "Minimal resource usage for embedded systems",
            },
            UseCase.RESEARCH_PROTOTYPE: {
                "optimize_for": "accuracy",
                "buffer_size": 16,
                "enable_fusion": True,
                "quantize": "none",
                "enable_pruning": False,
                "memory_limit": 512,
                "reasoning": "Maximum accuracy for research and development",
            },
            UseCase.PRODUCTION_DEPLOYMENT: {
                "optimize_for": "balanced",
                "buffer_size": 32,
                "enable_fusion": True,
                "quantize": "int8",
                "enable_pruning": True,
                "pruning_sparsity": 0.3,
                "memory_limit": 256,
                "reasoning": "Production-ready configuration with good performance",
            },
        }

    def _initialize_performance_profiles(
        self,
    ) -> Dict[PerformanceProfile, Dict[str, Any]]:
        """Initialize performance profiles for different optimization strategies."""
        return {
            PerformanceProfile.MAXIMUM_SPEED: {
                "optimize_for": "latency",
                "quantize": "int8",
                "enable_fusion": True,
                "enable_pruning": False,
                "buffer_size": 1,
                "priority": ["latency", "throughput"],
            },
            PerformanceProfile.MINIMUM_MEMORY: {
                "optimize_for": "memory",
                "quantize": "int8",
                "enable_fusion": True,
                "enable_pruning": True,
                "pruning_sparsity": 0.7,
                "buffer_size": 1,
                "priority": ["memory", "size"],
            },
            PerformanceProfile.MAXIMUM_ACCURACY: {
                "optimize_for": "accuracy",
                "quantize": "none",
                "enable_fusion": True,
                "enable_pruning": False,
                "buffer_size": 16,
                "priority": ["accuracy", "precision"],
            },
            PerformanceProfile.BALANCED: {
                "optimize_for": "balanced",
                "quantize": "int8",
                "enable_fusion": True,
                "enable_pruning": True,
                "pruning_sparsity": 0.3,
                "buffer_size": 16,
                "priority": ["latency", "memory", "accuracy"],
            },
            PerformanceProfile.ULTRA_LOW_POWER: {
                "optimize_for": "power",
                "quantize": "int8",
                "enable_fusion": False,
                "enable_pruning": True,
                "pruning_sparsity": 0.8,
                "buffer_size": 1,
                "priority": ["power", "memory"],
            },
        }

    def _initialize_device_optimizations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize device-specific optimization recommendations."""
        return {
            "raspberry_pi": {
                "recommended_quantize": "int8",
                "recommended_buffer_size": 16,
                "recommended_memory_limit": 256,
                "recommended_optimize_for": "latency",
                "avoid": ["float16", "large_buffer_sizes"],
                "prefer": ["int8", "fusion", "small_models"],
            },
            "jetson_nano": {
                "recommended_quantize": "float16",
                "recommended_buffer_size": 32,
                "recommended_memory_limit": 1024,
                "recommended_optimize_for": "balanced",
                "avoid": ["excessive_pruning"],
                "prefer": ["float16", "fusion", "moderate_pruning"],
            },
            "jetson_xavier": {
                "recommended_quantize": "float16",
                "recommended_buffer_size": 64,
                "recommended_memory_limit": 2048,
                "recommended_optimize_for": "accuracy",
                "avoid": ["int8_for_accuracy_critical"],
                "prefer": ["float16", "all_optimizations", "large_models"],
            },
            "cortex_m4": {
                "recommended_quantize": "int8",
                "recommended_buffer_size": 4,
                "recommended_memory_limit": 128,
                "recommended_optimize_for": "memory",
                "avoid": ["float16", "fusion", "large_buffer_sizes"],
                "prefer": ["int8", "pruning", "tiny_models"],
            },
            "cortex_m7": {
                "recommended_quantize": "int8",
                "recommended_buffer_size": 8,
                "recommended_memory_limit": 256,
                "recommended_optimize_for": "balanced",
                "avoid": ["float16", "large_buffer_sizes"],
                "prefer": ["int8", "fusion", "moderate_pruning"],
            },
            "cpu": {
                "recommended_quantize": "float16",
                "recommended_buffer_size": 64,
                "recommended_memory_limit": 1024,
                "recommended_optimize_for": "balanced",
                "avoid": ["excessive_optimization"],
                "prefer": ["float16", "fusion", "moderate_settings"],
            },
            "gpu": {
                "recommended_quantize": "float16",
                "recommended_buffer_size": 128,
                "recommended_memory_limit": 2048,
                "recommended_optimize_for": "accuracy",
                "avoid": ["int8_for_accuracy"],
                "prefer": ["float16", "all_optimizations", "large_models"],
            },
        }

    def suggest_configuration_improvements(
        self, config: Dict[str, Any]
    ) -> List[ConfigurationSuggestion]:
        """Suggest improvements for a given configuration.

        Args:
            config: Current configuration

        Returns:
            List of configuration suggestions
        """
        suggestions = []

        # Validate current configuration
        validation_result = self.validator.validate_config(config)

        # Generate suggestions based on validation issues
        for issue in validation_result.issues + validation_result.warnings:
            if issue.suggested_value is not None:
                suggestion = ConfigurationSuggestion(
                    parameter=issue.parameter or "unknown",
                    current_value=issue.current_value,
                    suggested_value=issue.suggested_value,
                    confidence=0.9 if issue.severity.value == "error" else 0.7,
                    reasoning=issue.explanation or "Validation issue detected",
                    impact=issue.impact or "Will improve configuration validity",
                )
                suggestions.append(suggestion)

        # Generate device-specific suggestions
        device = config.get("target_device", "cpu")
        if device in self.device_optimizations:
            device_recs = self.device_optimizations[device]
            suggestions.extend(self._generate_device_suggestions(config, device_recs))

        # Generate performance-based suggestions
        suggestions.extend(self._generate_performance_suggestions(config))

        # Generate use-case based suggestions
        suggestions.extend(self._generate_use_case_suggestions(config))

        return suggestions

    def _generate_device_suggestions(
        self, config: Dict[str, Any], device_recs: Dict[str, Any]
    ) -> List[ConfigurationSuggestion]:
        """Generate device-specific suggestions."""
        suggestions = []

        # Quantization suggestions
        current_quantize = config.get("quantize", "none")
        recommended_quantize = device_recs["recommended_quantize"]
        if current_quantize != recommended_quantize:
            suggestions.append(
                ConfigurationSuggestion(
                    parameter="quantize",
                    current_value=current_quantize,
                    suggested_value=recommended_quantize,
                    confidence=0.8,
                    reasoning=f"Device {config.get('target_device')} performs best with {recommended_quantize} quantization",
                    impact="Will improve performance and compatibility",
                    alternatives=[
                        ("none", "No quantization for maximum accuracy"),
                        ("int8", "INT8 quantization for size optimization"),
                    ],
                )
            )

        # Buffer size suggestions
        current_buffer = config.get("buffer_size", 1)
        recommended_buffer = device_recs["recommended_buffer_size"]
        if current_buffer != recommended_buffer:
            suggestions.append(
                ConfigurationSuggestion(
                    parameter="buffer_size",
                    current_value=current_buffer,
                    suggested_value=recommended_buffer,
                    confidence=0.7,
                    reasoning=f"Recommended buffer size for {config.get('target_device')} is {recommended_buffer}",
                    impact="Will optimize memory usage and performance",
                )
            )

        # Memory limit suggestions
        current_memory = config.get("memory_limit")
        recommended_memory = device_recs["recommended_memory_limit"]
        if current_memory and current_memory != recommended_memory:
            suggestions.append(
                ConfigurationSuggestion(
                    parameter="memory_limit",
                    current_value=current_memory,
                    suggested_value=recommended_memory,
                    confidence=0.6,
                    reasoning=f"Recommended memory limit for {config.get('target_device')} is {recommended_memory}MB",
                    impact="Will optimize memory allocation",
                )
            )

        return suggestions

    def _generate_performance_suggestions(
        self, config: Dict[str, Any]
    ) -> List[ConfigurationSuggestion]:
        """Generate performance-based suggestions."""
        suggestions = []

        optimize_for = config.get("optimize_for", "balanced")

        # Fusion suggestions based on optimization goal
        current_fusion = config.get("enable_fusion", True)
        if optimize_for == "latency" and not current_fusion:
            suggestions.append(
                ConfigurationSuggestion(
                    parameter="enable_fusion",
                    current_value=current_fusion,
                    suggested_value=True,
                    confidence=0.8,
                    reasoning="Fusion improves latency by reducing operation overhead",
                    impact="Will reduce inference latency by 10-20%",
                )
            )

        # Pruning suggestions based on optimization goal
        current_pruning = config.get("enable_pruning", False)
        if optimize_for == "memory" and not current_pruning:
            suggestions.append(
                ConfigurationSuggestion(
                    parameter="enable_pruning",
                    current_value=current_pruning,
                    suggested_value=True,
                    confidence=0.7,
                    reasoning="Pruning reduces model size and memory usage",
                    impact="Will reduce model size by 30-70%",
                )
            )

        # Buffer size suggestions based on optimization goal
        current_buffer = config.get("buffer_size", 1)
        if optimize_for == "latency" and current_buffer > 1:
            suggestions.append(
                ConfigurationSuggestion(
                    parameter="buffer_size",
                    current_value=current_buffer,
                    suggested_value=1,
                    confidence=0.6,
                    reasoning="Single buffer minimizes latency for real-time inference",
                    impact="Will reduce latency by eliminating buffering overhead",
                )
            )

        return suggestions

    def _generate_use_case_suggestions(
        self, config: Dict[str, Any]
    ) -> List[ConfigurationSuggestion]:
        """Generate use-case based suggestions."""
        suggestions = []

        # Detect use case based on configuration
        detected_use_case = self._detect_use_case(config)
        if detected_use_case:
            template = self.use_case_templates[detected_use_case]

            for param, recommended_value in template.items():
                if param == "reasoning":
                    continue

                current_value = config.get(param)
                if current_value != recommended_value:
                    suggestions.append(
                        ConfigurationSuggestion(
                            parameter=param,
                            current_value=current_value,
                            suggested_value=recommended_value,
                            confidence=0.6,
                            reasoning=f"Optimized for {detected_use_case.value} use case",
                            impact=f"Will improve {detected_use_case.value} performance",
                        )
                    )

        return suggestions

    def _detect_use_case(self, config: Dict[str, Any]) -> Optional[UseCase]:
        """Detect the most likely use case based on configuration."""
        buffer_size = config.get("buffer_size", 1)
        optimize_for = config.get("optimize_for", "balanced")
        memory_limit = config.get("memory_limit", 256)
        device = config.get("target_device", "cpu")

        # Real-time inference
        if buffer_size == 1 and optimize_for == "latency":
            return UseCase.REAL_TIME_INFERENCE

        # Batch processing
        if buffer_size > 32 and optimize_for == "memory":
            return UseCase.BATCH_PROCESSING

        # Mobile deployment
        if device in ["raspberry_pi", "jetson_nano"] and memory_limit < 512:
            return UseCase.MOBILE_DEPLOYMENT

        # IoT sensor
        if memory_limit < 128 and buffer_size <= 4:
            return UseCase.IOT_SENSOR

        # Edge server
        if memory_limit > 1024 and buffer_size > 64:
            return UseCase.EDGE_SERVER

        # Embedded system
        if device in ["cortex_m4", "cortex_m7"]:
            return UseCase.EMBEDDED_SYSTEM

        # Research prototype
        if optimize_for == "accuracy" and config.get("quantize", "none") == "none":
            return UseCase.RESEARCH_PROTOTYPE

        # Default to production deployment
        return UseCase.PRODUCTION_DEPLOYMENT

    def generate_optimized_config(
        self,
        base_config: Dict[str, Any],
        use_case: Optional[UseCase] = None,
        performance_profile: Optional[PerformanceProfile] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an optimized configuration based on requirements.

        Args:
            base_config: Base configuration to optimize
            use_case: Target use case
            performance_profile: Performance optimization profile
            device: Target device

        Returns:
            Optimized configuration
        """
        optimized_config = base_config.copy()

        # Apply device-specific optimizations
        if device and device in self.device_optimizations:
            device_recs = self.device_optimizations[device]
            optimized_config.update(
                {
                    "target_device": device,
                    "quantize": device_recs["recommended_quantize"],
                    "buffer_size": device_recs["recommended_buffer_size"],
                    "memory_limit": device_recs["recommended_memory_limit"],
                    "optimize_for": device_recs["recommended_optimize_for"],
                }
            )

        # Apply use case template
        if use_case and use_case in self.use_case_templates:
            template = self.use_case_templates[use_case]
            for param, value in template.items():
                if param != "reasoning":
                    optimized_config[param] = value

        # Apply performance profile
        if performance_profile and performance_profile in self.performance_profiles:
            profile = self.performance_profiles[performance_profile]
            for param, value in profile.items():
                if param != "priority":
                    optimized_config[param] = value

        return optimized_config

    def get_configuration_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration templates for common scenarios.

        Returns:
            Dictionary of template name -> configuration
        """
        templates = {}

        # Use case templates
        for use_case, config in self.use_case_templates.items():
            templates[f"use_case_{use_case.value}"] = {
                k: v for k, v in config.items() if k != "reasoning"
            }

        # Performance profile templates
        for profile, config in self.performance_profiles.items():
            templates[f"profile_{profile.value}"] = {
                k: v for k, v in config.items() if k != "priority"
            }

        # Device-specific templates
        for device, recs in self.device_optimizations.items():
            templates[f"device_{device}"] = {
                "target_device": device,
                "quantize": recs["recommended_quantize"],
                "buffer_size": recs["recommended_buffer_size"],
                "memory_limit": recs["recommended_memory_limit"],
                "optimize_for": recs["recommended_optimize_for"],
            }

        return templates

    def analyze_configuration_performance(
        self, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze configuration performance characteristics.

        Args:
            config: Configuration to analyze

        Returns:
            Performance analysis results
        """
        validation_result = self.validator.validate_config(config)

        # Detect use case
        use_case = self._detect_use_case(config)

        # Analyze optimization choices
        analysis = {
            "use_case": use_case.value if use_case else "unknown",
            "optimization_focus": config.get("optimize_for", "balanced"),
            "quantization_strategy": config.get("quantize", "none"),
            "memory_efficiency": "high"
            if config.get("memory_limit", 256) < 256
            else "medium",
            "latency_optimization": "high"
            if config.get("buffer_size", 1) == 1
            else "medium",
            "accuracy_preservation": "high"
            if config.get("quantize") == "none"
            else "medium",
            "compatibility_score": validation_result.compatibility_score,
            "estimated_performance": validation_result.estimated_performance_impact,
            "recommendations": [],
        }

        # Generate recommendations
        suggestions = self.suggest_configuration_improvements(config)
        for suggestion in suggestions:
            if suggestion.confidence > 0.7:
                analysis["recommendations"].append(
                    {
                        "parameter": suggestion.parameter,
                        "current": suggestion.current_value,
                        "suggested": suggestion.suggested_value,
                        "reasoning": suggestion.reasoning,
                        "impact": suggestion.impact,
                    }
                )

        return analysis


def suggest_configuration_improvements(
    config: Dict[str, Any]
) -> List[ConfigurationSuggestion]:
    """Suggest improvements for a configuration.

    Args:
        config: Current configuration

    Returns:
        List of configuration suggestions
    """
    suggester = EdgeFlowConfigSuggester()
    return suggester.suggest_configuration_improvements(config)


def generate_optimized_config(
    base_config: Dict[str, Any],
    use_case: Optional[UseCase] = None,
    performance_profile: Optional[PerformanceProfile] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate an optimized configuration.

    Args:
        base_config: Base configuration
        use_case: Target use case
        performance_profile: Performance profile
        device: Target device

    Returns:
        Optimized configuration
    """
    suggester = EdgeFlowConfigSuggester()
    return suggester.generate_optimized_config(
        base_config, use_case, performance_profile, device
    )


def get_configuration_templates() -> Dict[str, Dict[str, Any]]:
    """Get configuration templates.

    Returns:
        Dictionary of templates
    """
    suggester = EdgeFlowConfigSuggester()
    return suggester.get_configuration_templates()


def analyze_configuration_performance(config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze configuration performance.

    Args:
        config: Configuration to analyze

    Returns:
        Performance analysis
    """
    suggester = EdgeFlowConfigSuggester()
    return suggester.analyze_configuration_performance(config)


if __name__ == "__main__":
    # Test the config suggester
    test_config = {
        "model": "mobilenet_v2.tflite",
        "quantize": "float16",
        "target_device": "raspberry_pi",
        "enable_fusion": False,
        "buffer_size": 64,
        "memory_limit": 512,
        "optimize_for": "latency",
    }

    suggester = EdgeFlowConfigSuggester()

    print("=== Configuration Analysis ===")
    analysis = suggester.analyze_configuration_performance(test_config)
    for key, value in analysis.items():
        print(f"{key}: {value}")

    print("\n=== Configuration Suggestions ===")
    suggestions = suggester.suggest_configuration_improvements(test_config)
    for suggestion in suggestions:
        print(f"Parameter: {suggestion.parameter}")
        print(f"  Current: {suggestion.current_value}")
        print(f"  Suggested: {suggestion.suggested_value}")
        print(f"  Confidence: {suggestion.confidence:.2f}")
        print(f"  Reasoning: {suggestion.reasoning}")
        print(f"  Impact: {suggestion.impact}")
        print()

    print("=== Optimized Configuration ===")
    optimized = suggester.generate_optimized_config(
        test_config, use_case=UseCase.REAL_TIME_INFERENCE, device="raspberry_pi"
    )
    for key, value in optimized.items():
        print(f"{key} = {value}")
