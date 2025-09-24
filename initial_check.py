"""
EdgeFlow Initial Check Module

Validates model compatibility with target edge devices before optimization.
Determines if optimization is necessary based on device constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from device_specs import DeviceSpec, DeviceSpecManager

logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Profile of a machine learning model."""

    file_size_mb: float
    estimated_ram_mb: float
    num_parameters: int
    num_operations: int
    input_shape: List[int]
    output_shape: List[int]
    quantized: bool
    model_format: str


@dataclass
class CompatibilityReport:
    """Report of model-device compatibility."""

    compatible: bool
    requires_optimization: bool
    issues: List[str]
    recommendations: List[str]
    estimated_fit_score: float  # 0-100


class InitialChecker:
    """Performs initial compatibility checks for models and devices."""

    def __init__(self, device_spec_file: Optional[str] = None):
        """Initialize the checker with device specifications."""
        self.spec_manager = DeviceSpecManager(device_spec_file)

    @staticmethod
    def _is_quantized_by_name(model_path: str) -> bool:
        name = Path(model_path).name.lower()
        return any(k in name for k in ["int8", "quant", "uint8"]) or name.endswith(
            ".tflite"
        )

    def profile_model(self, model_path: str) -> ModelProfile:
        """Profile a model to extract its characteristics.

        This implementation avoids heavy ML deps and provides robust heuristics:
        - size-based parameter/ops estimation
        - name-based quantization detection
        """
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        size_bytes = p.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 6)

        # Heuristic estimates: assume float32 weights by default
        # Use 4 bytes/param unless name suggests int8
        quantized = self._is_quantized_by_name(model_path)
        bytes_per_param = 1 if quantized else 4
        num_params = max(int(size_bytes / max(bytes_per_param, 1)), 1)
        # Ops ~ 2x params as a rough upper bound
        num_ops = max(num_params * 2, 1)

        # Estimated peak RAM as: model size + 50% activation buffer
        estimated_ram_mb = round(size_mb * 1.5, 6)

        fmt = p.suffix.lower().lstrip(".") or "unknown"

        return ModelProfile(
            file_size_mb=size_mb,
            estimated_ram_mb=estimated_ram_mb,
            num_parameters=num_params,
            num_operations=num_ops,
            input_shape=[],
            output_shape=[],
            quantized=quantized,
            model_format=fmt,
        )

    def check_compatibility(
        self, model_path: str, target_device: str, config: Dict[str, Any]
    ) -> CompatibilityReport:
        """Check if a model is compatible with target device."""
        profile = self.profile_model(model_path)
        device = self.spec_manager.get_device_spec(target_device or "generic")

        # Use config memory_limit if specified, otherwise use device spec
        memory_limit_mb = float(config.get("memory_limit", device.ram_mb))
        max_model_size_mb = min(memory_limit_mb, device.max_model_size_mb)

        issues: List[str] = []
        recs: List[str] = []

        # Hard constraints
        if profile.file_size_mb > max_model_size_mb:
            issues.append(
                f"Model size {profile.file_size_mb:.2f}MB exceeds memory limit {max_model_size_mb}MB"
            )
            recs.append("Apply pruning/quantization to reduce model size")

        # RAM headroom constraint (reserve ~30% of available RAM for system)
        usable_ram_mb = max(int(memory_limit_mb * 0.7), 1)
        if profile.estimated_ram_mb > usable_ram_mb:
            issues.append(
                f"Estimated RAM {profile.estimated_ram_mb:.2f}MB exceeds safe budget {usable_ram_mb}MB"
            )
            recs.append("Reduce batch size or apply operator fusion to lower RAM usage")

        # Quantization preference/compatibility
        desired_quant = str(config.get("quantize", "none")).lower()
        if (
            desired_quant != "none"
            and desired_quant not in device.supported_quantizations
        ):
            issues.append(
                f"Device does not support requested quantization '{desired_quant}'"
            )
            recs.append(
                f"Choose one of supported quantizations: {', '.join(device.supported_quantizations)}"
            )

        # If model is unquantized and device strongly prefers int8 (e.g., Coral)
        if (
            (not profile.quantized)
            and ("int8" in device.supported_quantizations)
            and device.tpu_available
        ):
            recs.append("Quantize to int8 to leverage TPU acceleration")

        fit = self._calculate_fit_score(profile, device, memory_limit_mb)

        compatible = len(issues) == 0
        # Require optimization if not compatible or fit score is modest (<70)
        requires_opt = (
            (not compatible)
            or (fit < 70.0)
            or (desired_quant != "none" and not profile.quantized)
        )

        return CompatibilityReport(
            compatible=compatible,
            requires_optimization=requires_opt,
            issues=issues,
            recommendations=recs,
            estimated_fit_score=round(fit, 2),
        )

    def _calculate_fit_score(
        self, model_profile: ModelProfile, device_spec: DeviceSpec, memory_limit_mb: float
    ) -> float:
        """Calculate how well a model fits on a device (0-100)."""
        # Use config memory limit instead of device spec RAM
        max_model_size = min(memory_limit_mb, device_spec.max_model_size_mb)

        # Score components with simple caps
        size_score = max(
            0.0,
            min(
                100.0,
                (max_model_size / max(model_profile.file_size_mb, 1e-6))
                * 20.0,
            ),
        )
        # RAM: use safe budget of 70% of memory limit
        safe_ram = max(memory_limit_mb * 0.7, 1.0)
        ram_score = max(
            0.0,
            min(100.0, (safe_ram / max(model_profile.estimated_ram_mb, 1e-6)) * 40.0),
        )
        # Accelerator bonus
        accel_bonus = (
            10.0 if (device_spec.tpu_available or device_spec.gpu_available) else 0.0
        )
        # Baseline headroom
        base = 20.0
        score = min(100.0, base + size_score + ram_score + accel_bonus)
        return float(score)


def perform_initial_check(
    model_path: str, config: Dict[str, Any], device_spec_file: Optional[str] = None
) -> Tuple[bool, CompatibilityReport]:
    """
    Main entry point for initial compatibility check.

    Returns:
        Tuple of (should_proceed_with_optimization, compatibility_report)
    """
    target = str(config.get("target_device") or config.get("device") or "generic")
    checker = InitialChecker(device_spec_file)
    report = checker.check_compatibility(model_path, target, config)
    # Proceed with optimization if report suggests optimization is needed
    return report.requires_optimization, report
