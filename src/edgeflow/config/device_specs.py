"""
EdgeFlow Device Specifications Module

Manages device specifications and constraints for edge deployment validation.
This module provides the foundation for compatibility checking.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported edge device types."""

    RASPBERRY_PI_3 = "raspberry_pi_3"
    RASPBERRY_PI_4 = "raspberry_pi_4"
    RASPBERRY_PI_ZERO = "raspberry_pi_zero"
    JETSON_NANO = "jetson_nano"
    JETSON_TX2 = "jetson_tx2"
    JETSON_XAVIER = "jetson_xavier"
    CORAL_DEV_BOARD = "coral_dev_board"
    ARDUINO_NANO_33 = "arduino_nano_33"
    ESP32 = "esp32"
    GENERIC = "generic"


@dataclass
class DeviceSpec:
    """Device specification data class."""

    name: str
    device_type: DeviceType
    ram_mb: int
    storage_mb: int
    cpu_cores: int
    cpu_freq_mhz: int
    gpu_available: bool
    tpu_available: bool
    supported_operations: List[str]
    max_model_size_mb: int
    supported_quantizations: List[str]
    power_budget_watts: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "device_type": self.device_type.value,
            "ram_mb": self.ram_mb,
            "storage_mb": self.storage_mb,
            "cpu_cores": self.cpu_cores,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "gpu_available": self.gpu_available,
            "tpu_available": self.tpu_available,
            "supported_operations": list(self.supported_operations),
            "max_model_size_mb": self.max_model_size_mb,
            "supported_quantizations": list(self.supported_quantizations),
            "power_budget_watts": self.power_budget_watts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceSpec":
        """Create DeviceSpec from dictionary."""
        # Provide safe defaults where values are missing
        dt_val = str(data.get("device_type", DeviceType.GENERIC.value)).lower()
        try:
            dt = DeviceType(dt_val)
        except Exception:
            dt = DeviceType.GENERIC
        return cls(
            name=str(data.get("name", dt.value)),
            device_type=dt,
            ram_mb=int(data.get("ram_mb", 512)),
            storage_mb=int(data.get("storage_mb", 4096)),
            cpu_cores=int(data.get("cpu_cores", 1)),
            cpu_freq_mhz=int(data.get("cpu_freq_mhz", 800)),
            gpu_available=bool(data.get("gpu_available", False)),
            tpu_available=bool(data.get("tpu_available", False)),
            supported_operations=list(
                data.get("supported_operations", ["conv2d", "dense", "relu", "softmax"])
            ),
            max_model_size_mb=int(data.get("max_model_size_mb", 50)),
            supported_quantizations=[
                q.lower()
                for q in data.get(
                    "supported_quantizations", ["none", "float16", "int8"]
                )
            ],
            power_budget_watts=(
                float(data["power_budget_watts"])
                if data.get("power_budget_watts") is not None
                else None
            ),
        )


class DeviceSpecManager:
    """Manages loading and querying device specifications."""

    def __init__(self, spec_file: Optional[str] = None):
        """
        Initialize with device specifications.

        Args:
            spec_file: Path to CSV/JSON file with device specs
        """
        self.devices: Dict[str, DeviceSpec] = {}
        self._load_default_specs()
        if spec_file:
            self.load_custom_specs(spec_file)

    def _load_default_specs(self) -> None:
        """Load built-in device specifications."""

        # Define default specs for common devices
        def add(spec: DeviceSpec) -> None:
            self.devices[spec.name] = spec

        add(
            DeviceSpec(
                name="raspberry_pi_3",
                device_type=DeviceType.RASPBERRY_PI_3,
                ram_mb=1024,
                storage_mb=16000,
                cpu_cores=4,
                cpu_freq_mhz=1200,
                gpu_available=False,
                tpu_available=False,
                supported_operations=[
                    "conv2d",
                    "depthwise_conv2d",
                    "dense",
                    "relu",
                    "softmax",
                ],
                max_model_size_mb=50,
                supported_quantizations=["none", "float16", "int8"],
                power_budget_watts=5.0,
            )
        )
        add(
            DeviceSpec(
                name="raspberry_pi_4",
                device_type=DeviceType.RASPBERRY_PI_4,
                ram_mb=2048,
                storage_mb=32000,
                cpu_cores=4,
                cpu_freq_mhz=1500,
                gpu_available=False,
                tpu_available=False,
                supported_operations=[
                    "conv2d",
                    "dense",
                    "relu",
                    "softmax",
                    "pad",
                    "add",
                ],
                max_model_size_mb=100,
                supported_quantizations=["none", "float16", "int8"],
                power_budget_watts=7.0,
            )
        )
        add(
            DeviceSpec(
                name="raspberry_pi_zero",
                device_type=DeviceType.RASPBERRY_PI_ZERO,
                ram_mb=512,
                storage_mb=8000,
                cpu_cores=1,
                cpu_freq_mhz=1000,
                gpu_available=False,
                tpu_available=False,
                supported_operations=["conv2d", "dense", "relu", "softmax"],
                max_model_size_mb=20,
                supported_quantizations=["float16", "int8"],
                power_budget_watts=2.5,
            )
        )
        add(
            DeviceSpec(
                name="jetson_nano",
                device_type=DeviceType.JETSON_NANO,
                ram_mb=4096,
                storage_mb=64000,
                cpu_cores=4,
                cpu_freq_mhz=1479,
                gpu_available=True,
                tpu_available=False,
                supported_operations=[
                    "conv2d",
                    "dense",
                    "relu",
                    "softmax",
                    "batchnorm",
                    "add",
                ],
                max_model_size_mb=200,
                supported_quantizations=["none", "float16", "int8"],
                power_budget_watts=10.0,
            )
        )
        add(
            DeviceSpec(
                name="jetson_tx2",
                device_type=DeviceType.JETSON_TX2,
                ram_mb=8192,
                storage_mb=128000,
                cpu_cores=6,
                cpu_freq_mhz=2035,
                gpu_available=True,
                tpu_available=False,
                supported_operations=[
                    "conv2d",
                    "dense",
                    "relu",
                    "softmax",
                    "add",
                    "mul",
                ],
                max_model_size_mb=400,
                supported_quantizations=["none", "float16", "int8"],
            )
        )
        add(
            DeviceSpec(
                name="jetson_xavier",
                device_type=DeviceType.JETSON_XAVIER,
                ram_mb=16384,
                storage_mb=256000,
                cpu_cores=8,
                cpu_freq_mhz=2300,
                gpu_available=True,
                tpu_available=False,
                supported_operations=[
                    "conv2d",
                    "dense",
                    "relu",
                    "softmax",
                    "add",
                    "mul",
                    "pad",
                ],
                max_model_size_mb=800,
                supported_quantizations=["none", "float16", "int8"],
            )
        )
        add(
            DeviceSpec(
                name="coral_dev_board",
                device_type=DeviceType.CORAL_DEV_BOARD,
                ram_mb=1024,
                storage_mb=8000,
                cpu_cores=4,
                cpu_freq_mhz=1500,
                gpu_available=False,
                tpu_available=True,
                supported_operations=["conv2d", "dense", "relu", "softmax"],
                max_model_size_mb=50,
                supported_quantizations=["int8"],
                power_budget_watts=5.0,
            )
        )
        add(
            DeviceSpec(
                name="arduino_nano_33",
                device_type=DeviceType.ARDUINO_NANO_33,
                ram_mb=1,
                storage_mb=1024,
                cpu_cores=1,
                cpu_freq_mhz=64,
                gpu_available=False,
                tpu_available=False,
                supported_operations=["dense", "relu", "softmax"],
                max_model_size_mb=1,
                supported_quantizations=["int8"],
                power_budget_watts=0.5,
            )
        )
        add(
            DeviceSpec(
                name="esp32",
                device_type=DeviceType.ESP32,
                ram_mb=0,
                storage_mb=4096,
                cpu_cores=2,
                cpu_freq_mhz=240,
                gpu_available=False,
                tpu_available=False,
                supported_operations=["dense", "relu", "softmax"],
                max_model_size_mb=2,
                supported_quantizations=["int8"],
                power_budget_watts=1.5,
            )
        )
        add(self._get_generic_spec())

    def _get_generic_spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="generic",
            device_type=DeviceType.GENERIC,
            ram_mb=2048,
            storage_mb=16000,
            cpu_cores=2,
            cpu_freq_mhz=1400,
            gpu_available=False,
            tpu_available=False,
            supported_operations=["conv2d", "dense", "relu", "softmax", "add"],
            max_model_size_mb=100,
            supported_quantizations=["none", "float16", "int8"],
        )

    def _load_csv_specs(self, path: Path) -> None:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("name"):
                    logger.warning("Skipping device row without name in %s", path)
                    continue
                # Parse list-like columns
                ops = row.get("supported_operations") or ""
                quants = row.get("supported_quantizations") or ""
                ops_list = [
                    x.strip() for x in ops.replace("|", ",").split(",") if x.strip()
                ]
                quants_list = [
                    x.strip().lower()
                    for x in quants.replace("|", ",").split(",")
                    if x.strip()
                ]
                data: Dict[str, Any] = {
                    **row,
                    "ram_mb": int(row.get("ram_mb") or 0),
                    "storage_mb": int(row.get("storage_mb") or 0),
                    "cpu_cores": int(row.get("cpu_cores") or 1),
                    "cpu_freq_mhz": int(row.get("cpu_freq_mhz") or 0),
                    "gpu_available": str(row.get("gpu_available", "")).lower()
                    in {"1", "true", "yes"},
                    "tpu_available": str(row.get("tpu_available", "")).lower()
                    in {"1", "true", "yes"},
                    "max_model_size_mb": int(row.get("max_model_size_mb") or 0),
                    "supported_operations": ops_list or ["conv2d", "dense"],
                    "supported_quantizations": quants_list
                    or ["none", "float16", "int8"],
                }
                spec = DeviceSpec.from_dict(data)
                self.devices[spec.name] = spec

    def _load_json_specs(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "devices" in data:
            items = data.get("devices", [])
        elif isinstance(data, list):
            items = data
        else:
            items = []
        for item in items:
            try:
                spec = DeviceSpec.from_dict(item)
                self.devices[spec.name] = spec
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping invalid device spec in %s: %s", path, exc)

    def load_custom_specs(self, spec_file: str) -> None:
        """Load custom device specifications from file."""
        path = Path(spec_file)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() == ".csv":
            self._load_csv_specs(path)
        elif path.suffix.lower() == ".json":
            self._load_json_specs(path)
        else:
            raise ValueError(f"Unsupported spec file format: {path.suffix}")

    def get_device_spec(self, device_name: str) -> DeviceSpec:
        """Get specification for a specific device."""
        key = (device_name or "").strip().lower()
        if not key:
            return self._get_generic_spec()
        # Direct hit
        if key in self.devices:
            return self.devices[key]
        # Try mapping from known aliases
        for name, spec in self.devices.items():
            if name.lower() == key or spec.device_type.value == key:
                return spec
        logger.warning("Unknown device: %s, using generic spec", device_name)
        return self._get_generic_spec()
