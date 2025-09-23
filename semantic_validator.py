"""Semantic validation passes for EdgeFlow IR/config.

This module provides early semantic checks and diagnostics scaffolding.
It operates on the parsed configuration dictionary for now and can be
extended to validate a richer IR graph when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Diagnostic:
    code: str
    severity: str  # error | warning | info
    message: str
    hint: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class SemanticValidator:
    def __init__(self, device_registry: Optional[Dict[str, Any]] = None) -> None:
        self.device_registry = device_registry or {}

    def validate_config(self, config: Dict[str, Any]) -> List[Diagnostic]:
        diagnostics: List[Diagnostic] = []

        # Example: simple rule stubs; extend with real IR-based checks later
        if "quantize" in config and config.get("quantize") == "int8":
            # Require a model present to be meaningful
            if "model" not in config:
                diagnostics.append(
                    Diagnostic(
                        code="EF101",
                        severity="error",
                        message="Quantization requires a model to be specified",
                        hint='Add `model = "path/to/file.tflite"`',
                        context={"parameter": "quantize"},
                    )
                )

        # Device presence
        if "target_device" in config:
            device = str(config["target_device"]).lower()
            if self.device_registry and device not in self.device_registry:
                diagnostics.append(
                    Diagnostic(
                        code="EF401",
                        severity="error",
                        message=f"Unsupported target device: {device}",
                        hint="Use a supported device or omit target_device",
                        context={"parameter": "target_device"},
                    )
                )

        return diagnostics
