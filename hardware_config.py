"""
EdgeFlow Hardware Configuration Module

Provides hardware configuration management and model compatibility validation
for edge deployment targets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from device_specs import DeviceSpec, DeviceSpecManager

logger = logging.getLogger(__name__)


class HardwareConfig:
    """Hardware configuration for EdgeFlow deployments."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize hardware configuration.

        Args:
            config_file: Path to hardware configuration JSON file
        """
        self.device_manager = DeviceSpecManager()
        self.custom_configs: Dict[str, Dict[str, Any]] = {}

        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file: str) -> None:
        """Load hardware configuration from JSON file."""
        path = Path(config_file)
        if not path.exists():
            logger.warning("Hardware config file not found: %s", config_file)
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Load custom device specs if provided
            if "device_specs_file" in config:
                specs_file = config["device_specs_file"]
                if Path(specs_file).exists():
                    self.device_manager.load_custom_specs(specs_file)
                else:
                    logger.warning("Device specs file not found: %s", specs_file)

            # Store custom configurations
            self.custom_configs = config.get("custom_configs", {})

            logger.info("Loaded hardware configuration from %s", config_file)

        except Exception as e:
            logger.error("Failed to load hardware config: %s", e)

    def validate_model_compatibility(
        self,
        model_path: str,
        target_device: str,
        config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate if a model is compatible with the target device.

        Args:
            model_path: Path to the model file
            target_device: Target device name
            config: EdgeFlow configuration dictionary

        Returns:
            Tuple of (is_compatible, list_of_warnings/issues)
        """
        warnings = []
        device_spec = self.device_manager.get_device_spec(target_device)

        # Check model file size
        try:
            model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            if model_size_mb > device_spec.max_model_size_mb:
                warnings.append(
                    f"Model size ({model_size_mb:.1f}MB) exceeds device limit "
                    f"({device_spec.max_model_size_mb}MB) for {target_device}"
                )
        except Exception as e:
            warnings.append(f"Could not check model size: {e}")

        # Check quantization compatibility
        quantize = str(config.get("quantize", "none")).lower()
        if quantize not in device_spec.supported_quantizations:
            warnings.append(
                f"Quantization '{quantize}' not supported by {target_device}. "
                f"Supported: {', '.join(device_spec.supported_quantizations)}"
            )

        # Check memory limit
        memory_limit = config.get("memory_limit", 0)
        if memory_limit > device_spec.ram_mb:
            warnings.append(
                f"Memory limit ({memory_limit}MB) exceeds device RAM "
                f"({device_spec.ram_mb}MB) for {target_device}"
            )

        # Check framework compatibility
        framework = config.get("framework", "tensorflow")
        # Most devices support TensorFlow, but some have specific requirements
        if device_spec.device_type.value == "coral_dev_board" and framework != "tensorflow":
            warnings.append("Coral Dev Board requires TensorFlow models")

        # Check power budget if specified
        if device_spec.power_budget_watts and config.get("power_aware", False):
            # This could be extended with power consumption estimates
            pass

        # Check for known device-specific constraints
        device_warnings = self._check_device_specific_constraints(
            device_spec, config
        )
        warnings.extend(device_warnings)

        is_compatible = len(warnings) == 0
        return is_compatible, warnings

    def _check_device_specific_constraints(
        self, device_spec: DeviceSpec, config: Dict[str, Any]
    ) -> List[str]:
        """Check device-specific constraints and return warnings."""
        warnings = []

        device_type = device_spec.device_type.value

        if device_type.startswith("raspberry_pi"):
            # Raspberry Pi specific checks
            if config.get("enable_pruning", False) and device_spec.ram_mb < 1024:
                warnings.append(
                    "Pruning may be slow on low-memory Raspberry Pi devices"
                )

        elif device_type.startswith("jetson"):
            # Jetson specific checks
            if not device_spec.gpu_available:
                warnings.append("Jetson device should have GPU available")

        elif device_type == "coral_dev_board":
            # Coral TPU specific checks
            if config.get("quantize", "none") != "int8":
                warnings.append("Coral TPU requires INT8 quantization")

        elif device_type in ["arduino_nano_33", "esp32"]:
            # Microcontroller specific checks
            if config.get("quantize", "none") != "int8":
                warnings.append("Microcontrollers require INT8 quantization")

            model_size = config.get("estimated_model_size_mb", 0)
            if model_size > device_spec.max_model_size_mb:
                warnings.append(
                    f"Model too large for microcontroller "
                    f"({model_size}MB > {device_spec.max_model_size_mb}MB)"
                )

        return warnings

    def get_device_recommendations(
        self,
        model_size_mb: float,
        requirements: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """
        Get device recommendations based on model size and requirements.

        Args:
            model_size_mb: Model size in MB
            requirements: Dictionary of requirements (e.g., {"gpu": True})

        Returns:
            List of (device_name, reason) tuples
        """
        recommendations = []

        for device_name, spec in self.device_manager.devices.items():
            if model_size_mb <= spec.max_model_size_mb:
                reasons = []

                # Check GPU requirement
                gpu_required = requirements.get("gpu", None)
                if gpu_required is True and not spec.gpu_available:
                    continue  # Skip if GPU required but not available
                elif gpu_required is False and spec.gpu_available:
                    continue  # Skip if GPU not wanted but device has GPU

                # Check TPU requirement
                tpu_required = requirements.get("tpu", None)
                if tpu_required is True and not spec.tpu_available:
                    continue  # Skip if TPU required but not available
                elif tpu_required is False and spec.tpu_available:
                    continue  # Skip if TPU not wanted but device has TPU

                # Build reasons for recommendation
                reasons = []

                # Check power budget requirement
                low_power_required = requirements.get("low_power", False)
                if low_power_required:
                    if spec.power_budget_watts and spec.power_budget_watts <= 5.0:
                        reasons.append("low power consumption")
                    else:
                        continue  # Skip devices that don't meet low power requirement

                # Add capability reasons if no specific requirements
                if not any(requirements.values()):
                    if spec.gpu_available:
                        reasons.append("GPU capable")
                    if spec.tpu_available:
                        reasons.append("TPU capable")
                    if spec.power_budget_watts and spec.power_budget_watts <= 5.0:
                        reasons.append("energy efficient")

                if reasons:
                    reason_str = ", ".join(reasons)
                    recommendations.append((device_name, reason_str))
                else:
                    recommendations.append((device_name, "fits model size"))

        # Sort by model size headroom (devices with more capacity first)
        recommendations.sort(
            key=lambda x: self.device_manager.devices[x[0]].max_model_size_mb - model_size_mb,
            reverse=True
        )

        return recommendations[:5]  # Return top 5 recommendations

    def get_optimization_suggestions(
        self,
        target_device: str,
        current_config: Dict[str, Any]
    ) -> List[str]:
        """
        Get optimization suggestions for a specific device.

        Args:
            target_device: Target device name
            current_config: Current EdgeFlow configuration

        Returns:
            List of optimization suggestions
        """
        suggestions = []
        device_spec = self.device_manager.get_device_spec(target_device)

        # Memory-based suggestions
        if device_spec.ram_mb < 2048:
            suggestions.append("Use INT8 quantization to reduce memory usage")
            suggestions.append("Enable operator fusion for better memory efficiency")

        # Compute capability suggestions
        if device_spec.cpu_cores >= 4:
            suggestions.append("Consider batch processing for better CPU utilization")
        elif device_spec.cpu_cores == 1:
            suggestions.append("Optimize for single-threaded execution")

        # Device-specific suggestions
        if device_spec.tpu_available:
            suggestions.append("Use Edge TPU for maximum performance")
        elif device_spec.gpu_available:
            suggestions.append("Enable GPU acceleration if available")

        # Power-aware suggestions
        if device_spec.power_budget_watts and device_spec.power_budget_watts <= 5.0:
            suggestions.append("Use power-efficient quantization (INT8)")
            suggestions.append("Consider model pruning for reduced computation")

        return suggestions

    def export_device_specs(self, output_file: str) -> None:
        """Export all device specifications to JSON file."""
        devices_data = {
            "devices": [spec.to_dict() for spec in self.device_manager.devices.values()]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(devices_data, f, indent=2)

        logger.info("Exported device specs to %s", output_file)


# Global hardware configuration instance
_hardware_config: Optional[HardwareConfig] = None


def get_hardware_config(config_file: Optional[str] = None) -> HardwareConfig:
    """Get or create global hardware configuration instance."""
    global _hardware_config
    if _hardware_config is None:
        _hardware_config = HardwareConfig(config_file)
    return _hardware_config


def validate_model_for_device(
    model_path: str,
    target_device: str,
    config: Dict[str, Any],
    hardware_config_file: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate model compatibility with a device.

    Args:
        model_path: Path to the model file
        target_device: Target device name
        config: EdgeFlow configuration
        hardware_config_file: Optional hardware config file

    Returns:
        Tuple of (is_compatible, list_of_warnings)
    """
    hw_config = get_hardware_config(hardware_config_file)
    return hw_config.validate_model_compatibility(model_path, target_device, config)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hardware_config.py <command> [args...]")
        print("Commands:")
        print("  export <output_file>    - Export device specs to JSON")
        print("  validate <model_path> <device> <config_file> - Validate model for device")
        print("  recommend <model_size_mb> [requirements] - Get device recommendations")
        print("  test                     - Run basic tests")
        sys.exit(1)

    command = sys.argv[1]

    if command == "export":
        if len(sys.argv) < 3:
            print("Usage: python hardware_config.py export <output_file>")
            sys.exit(1)
        output_file = sys.argv[2]
        config = HardwareConfig()
        config.export_device_specs(output_file)
        print(f"Device specs exported to {output_file}")

    elif command == "validate":
        if len(sys.argv) < 5:
            print("Usage: python hardware_config.py validate <model_path> <device> <config_file>")
            sys.exit(1)
        model_path = sys.argv[2]
        device = sys.argv[3]
        config_file = sys.argv[4]

        # Load config from file
        import json
        with open(config_file, 'r') as f:
            test_config = json.load(f)

        config = HardwareConfig()
        is_compatible, warnings = config.validate_model_compatibility(
            model_path, device, test_config
        )

        print(f"Model compatibility check for {model_path} on {device}:")
        print(f"Compatible: {is_compatible}")
        if warnings:
            print("Issues/Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("No issues found!")

    elif command == "recommend":
        if len(sys.argv) < 3:
            print("Usage: python hardware_config.py recommend <model_size_mb> [gpu=true] [tpu=true] [low_power=true]")
            sys.exit(1)

        model_size = float(sys.argv[2])
        requirements = {}

        for arg in sys.argv[3:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                requirements[key] = value.lower() in ("true", "1", "yes")

        config = HardwareConfig()
        recommendations = config.get_device_recommendations(model_size, requirements)

        print(f"Device recommendations for {model_size}MB model:")
        for device, reason in recommendations:
            print(f"  - {device}: {reason}")

    elif command == "test":
        # Run basic tests
        config = HardwareConfig()

        # Export device specs
        config.export_device_specs("device_specs_test.json")
        print("✓ Exported device specs")

        # Test validation
        test_config = {
            "quantize": "int8",
            "memory_limit": 64,
            "enable_pruning": True
        }

        is_compatible, warnings = config.validate_model_compatibility(
            "large_resnet_model.h5", "raspberry_pi_4", test_config
        )

        print(f"✓ Validation test: Compatible={is_compatible}, Warnings={len(warnings)}")

        # Get recommendations
        recommendations = config.get_device_recommendations(25.0, {"gpu": False})
        print(f"✓ Got {len(recommendations)} device recommendations")

        # Get optimization suggestions
        suggestions = config.get_optimization_suggestions("raspberry_pi_zero", test_config)
        print(f"✓ Got {len(suggestions)} optimization suggestions")

        print("All tests passed!")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)