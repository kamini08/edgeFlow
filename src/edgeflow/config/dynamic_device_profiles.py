"""EdgeFlow Dynamic Device Profile Management System

This module provides dynamic management of device profiles with runtime updates,
automatic profile discovery, and configurable constraint management for new hardware.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class DeviceCategory(Enum):
    """Categories of devices for grouping similar hardware."""

    EDGE = "edge"
    MOBILE = "mobile"
    EMBEDDED = "embedded"
    SERVER = "server"
    CLOUD = "cloud"
    CUSTOM = "custom"


class ConstraintType(Enum):
    """Types of constraints that can be applied to devices."""

    MEMORY = "memory"
    COMPUTE = "compute"
    POWER = "power"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    LATENCY = "latency"
    THERMAL = "thermal"


@dataclass
class DeviceConstraint:
    """A single constraint for a device."""

    constraint_type: ConstraintType
    name: str
    value: Union[int, float, str]
    unit: str
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    description: str = ""
    enforced: bool = True

    def validate_value(self, test_value: Union[int, float]) -> bool:
        """Validate a value against this constraint."""
        if not self.enforced:
            return True

        if isinstance(self.value, (int, float)):
            if self.min_value is not None and test_value < self.min_value:
                return False
            if self.max_value is not None and test_value > self.max_value:
                return False
            return test_value <= self.value

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceConstraint":
        """Create from dictionary."""
        data["constraint_type"] = ConstraintType(data["constraint_type"])
        return cls(**data)


class DeviceCapability:
    """A capability or feature of a device.

    Args:
        name: Name of the capability
        supported: Whether the capability is supported
        version: Version of the capability if applicable
        performance_score: Relative performance score (0.0-1.0)
        description: Human-readable description of the capability
        **metadata: Additional metadata about the capability
    """

    def __init__(
        self,
        name: str,
        supported: bool,
        version: Optional[str] = None,
        performance_score: Optional[float] = None,
        description: Optional[str] = None,
        **metadata,
    ):
        self.name = name
        self.supported = supported
        self.version = version
        self.performance_score = performance_score
        self.metadata = metadata.copy()
        if description is not None:
            self.metadata["description"] = description

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "supported": self.supported,
            "version": self.version,
            "performance_score": self.performance_score,
            **self.metadata,
        }
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceCapability":
        """Create from dictionary."""
        # Extract known fields
        known_fields = {"name", "supported", "version", "performance_score"}
        metadata = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            name=data["name"],
            supported=data["supported"],
            version=data.get("version"),
            performance_score=data.get("performance_score"),
            **metadata,
        )


@dataclass
class DeviceProfile:
    """Complete profile for a device including constraints and capabilities."""

    device_id: str
    name: str
    category: DeviceCategory
    manufacturer: str = ""
    model: str = ""
    version: str = ""

    # Hardware specifications
    constraints: List[DeviceConstraint] = field(default_factory=list)
    capabilities: List[DeviceCapability] = field(default_factory=list)

    # Optimization hints
    preferred_quantization: List[str] = field(default_factory=lambda: ["int8"])
    supported_operators: Set[str] = field(default_factory=set)
    optimization_flags: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    description: str = ""

    # Performance benchmarks
    benchmark_scores: Dict[str, float] = field(default_factory=dict)

    def add_constraint(
        self,
        constraint_type: ConstraintType,
        name: str,
        value: Union[int, float, str],
        unit: str,
        **kwargs,
    ) -> None:
        """Add a constraint to the device profile."""
        constraint = DeviceConstraint(
            constraint_type=constraint_type, name=name, value=value, unit=unit, **kwargs
        )
        self.constraints.append(constraint)
        self.updated_at = datetime.now()

    def add_capability(
        self, name: str, supported: bool, version: Optional[str] = None, **kwargs
    ) -> None:
        """Add a capability to the device profile.

        Args:
            name: Name of the capability
            supported: Whether the capability is supported
            version: Version of the capability if applicable
            **kwargs: Additional metadata including 'description', 'performance_score', etc.
        """
        # Extract known fields
        performance_score = kwargs.pop("performance_score", None)

        capability = DeviceCapability(
            name=name,
            supported=supported,
            version=version,
            performance_score=performance_score,
            **kwargs,
        )
        self.capabilities.append(capability)
        self.updated_at = datetime.now()

    def get_constraint(self, name: str) -> Optional[DeviceConstraint]:
        """Get a constraint by name."""
        for constraint in self.constraints:
            if constraint.name == name:
                return constraint
        return None

    def get_capability(self, name: str) -> Optional[DeviceCapability]:
        """Get a capability by name."""
        for capability in self.capabilities:
            if capability.name == name:
                return capability
        return None

    def validate_model_requirements(self, requirements: Dict[str, Any]) -> bool:
        """Validate model requirements against device constraints.

        Args:
            requirements: Dictionary of requirement names and their required values

        Returns:
            bool: True if all requirements are satisfied, False otherwise
        """
        for req_name, req_value in requirements.items():
            constraint = self.get_constraint(req_name)
            if not constraint:
                return False

            if constraint.min_value is not None and req_value < constraint.min_value:
                return False

            if constraint.max_value is not None and req_value > constraint.max_value:
                return False

        return True

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get recommended optimization configuration for this device.

        Returns:
            Dict containing optimization settings including:
            - quantization: List of supported quantization types
            - preferred_precision: Recommended precision for this device (e.g., 'int8', 'fp16')
            - enabled_optimizations: List of enabled optimization strategies
            - optimization_flags: Dictionary of optimization flags
            - target_device: ID of the target device
            - enable_*: Various optimization toggles
        """
        config: Dict[str, Any] = {
            "quantization": self.preferred_quantization,
            "preferred_precision": (
                "int8" if "int8" in self.preferred_quantization else "fp16"
            ),
            "enabled_optimizations": [],
            "optimization_flags": {},
            "target_device": self.device_id,
        }

        # Add device-specific optimizations based on constraints
        memory_constraint = self.get_constraint(
            "total_memory_mb"
        ) or self.get_constraint("max_memory_mb")
        if (
            memory_constraint
            and isinstance(memory_constraint.value, (int, float))
            and memory_constraint.value < 2048
        ):  # Less than 2GB
            config["enable_aggressive_pruning"] = True
            config["enable_memory_optimization"] = True
            config["optimization_flags"]["reduce_memory_footprint"] = True
            config["enabled_optimizations"].extend(["pruning", "memory_optimization"])

            # Prefer more aggressive quantization for low-memory devices
            if "int8" in self.preferred_quantization:
                config["preferred_precision"] = "int8"

        # Check for compute constraints
        cpu_cores = self.get_constraint("cpu_cores")
        if (
            cpu_cores
            and isinstance(cpu_cores.value, (int, float))
            and cpu_cores.value <= 4
        ):
            config["optimization_flags"]["optimize_for_low_core_count"] = True
            config["enabled_optimizations"].append("low_core_optimization")

            # For low-core devices, prefer lower precision
            if "int8" in self.preferred_quantization:
                config["preferred_precision"] = "int8"

        # Add capabilities
        gpu_cap = self.get_capability("gpu_acceleration")
        if gpu_cap and gpu_cap.supported:
            config["enable_gpu"] = True
            config["gpu_backend"] = "cuda"  # Default, can be overridden

            # If GPU is available, we can use mixed precision
            config["mixed_precision"] = True
            config["enabled_optimizations"].append("gpu_acceleration")

            if "fp16" in self.preferred_quantization:
                config["preferred_precision"] = "fp16"

        # Add supported operators if available
        if hasattr(self, "supported_operators") and self.supported_operators:
            config["supported_operators"] = list(self.supported_operators)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "category": self.category.value,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "version": self.version,
            "constraints": [c.to_dict() for c in self.constraints],
            "capabilities": [c.to_dict() for c in self.capabilities],
            "preferred_quantization": self.preferred_quantization,
            "supported_operators": list(self.supported_operators),
            "optimization_flags": self.optimization_flags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags,
            "description": self.description,
            "benchmark_scores": self.benchmark_scores,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceProfile":
        """Create from dictionary."""
        # Convert datetime strings back to datetime objects
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["category"] = DeviceCategory(data["category"])

        # Convert constraints and capabilities
        data["constraints"] = [
            DeviceConstraint.from_dict(c) for c in data["constraints"]
        ]
        data["capabilities"] = [
            DeviceCapability.from_dict(c) for c in data["capabilities"]
        ]
        data["supported_operators"] = set(data["supported_operators"])

        return cls(**data)


class DeviceProfileManager:
    """Manages device profiles with dynamic updates and discovery."""

    def __init__(self, profiles_dir: str = "device_profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self.profiles: Dict[str, DeviceProfile] = {}
        self.profile_cache: Dict[str, float] = {}  # device_id -> last_modified_time

        # Load existing profiles
        self._load_all_profiles()

        # Create default profiles if none exist
        if not self.profiles:
            self._create_default_profiles()

    def _load_all_profiles(self) -> None:
        """Load all profiles from the profiles directory."""
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                self._load_profile_file(profile_file)
            except Exception as e:
                logger.error(f"Failed to load profile {profile_file}: {e}")

    def _load_profile_file(self, profile_file: Path) -> None:
        """Load a single profile file."""
        with open(profile_file, "r") as f:
            data = json.load(f)

        profile = DeviceProfile.from_dict(data)
        self.profiles[profile.device_id] = profile
        self.profile_cache[profile.device_id] = profile_file.stat().st_mtime

        logger.debug(f"Loaded device profile: {profile.name} ({profile.device_id})")

    def _create_default_profiles(self) -> None:
        """Create default device profiles."""
        logger.info("Creating default device profiles...")

        # Raspberry Pi 4
        rpi4 = DeviceProfile(
            device_id="raspberry_pi_4",
            name="Raspberry Pi 4",
            category=DeviceCategory.EDGE,
            manufacturer="Raspberry Pi Foundation",
            model="4B",
            description="Popular single-board computer for edge AI",
        )
        rpi4.add_constraint(
            ConstraintType.MEMORY,
            "max_memory_mb",
            4096,
            "MB",
            description="Maximum available RAM",
        )
        rpi4.add_constraint(
            ConstraintType.COMPUTE,
            "max_ops_per_sec",
            5e8,
            "ops/sec",
            description="Approximate compute capacity",
        )
        rpi4.add_constraint(
            ConstraintType.POWER,
            "max_power_watts",
            15,
            "W",
            description="Maximum power consumption",
        )
        rpi4.add_capability("arm_neon", True, description="ARM NEON SIMD instructions")
        rpi4.add_capability(
            "gpu_acceleration",
            True,
            version="VideoCore VI",
            description="GPU acceleration support",
        )
        rpi4.preferred_quantization = ["int8", "float16"]
        rpi4.optimization_flags = {"enable_neon": True, "enable_gpu_delegate": True}

        # Mobile device (generic)
        mobile = DeviceProfile(
            device_id="mobile_generic",
            name="Generic Mobile Device",
            category=DeviceCategory.MOBILE,
            description="Typical smartphone/tablet specifications",
        )
        mobile.add_constraint(ConstraintType.MEMORY, "max_memory_mb", 6144, "MB")
        mobile.add_constraint(ConstraintType.COMPUTE, "max_ops_per_sec", 2e9, "ops/sec")
        mobile.add_constraint(ConstraintType.POWER, "max_power_watts", 5, "W")
        mobile.add_constraint(ConstraintType.LATENCY, "max_inference_ms", 100, "ms")
        mobile.add_capability(
            "gpu_acceleration", True, description="Mobile GPU support"
        )
        mobile.add_capability("dsp_acceleration", True, description="DSP acceleration")
        mobile.preferred_quantization = ["int8", "float16"]

        # Edge TPU
        edge_tpu = DeviceProfile(
            device_id="coral_edge_tpu",
            name="Coral Edge TPU",
            category=DeviceCategory.EDGE,
            manufacturer="Google",
            description="Specialized AI accelerator for edge inference",
        )
        edge_tpu.add_constraint(ConstraintType.MEMORY, "max_memory_mb", 8192, "MB")
        edge_tpu.add_constraint(
            ConstraintType.COMPUTE, "max_ops_per_sec", 4e12, "ops/sec"
        )
        edge_tpu.add_constraint(ConstraintType.POWER, "max_power_watts", 2, "W")
        edge_tpu.add_capability(
            "tpu_acceleration", True, version="Edge TPU", performance_score=95.0
        )
        edge_tpu.preferred_quantization = ["int8"]
        edge_tpu.optimization_flags = {"require_tpu_compatible": True}

        # Server GPU
        server_gpu = DeviceProfile(
            device_id="server_gpu_v100",
            name="NVIDIA V100 Server",
            category=DeviceCategory.SERVER,
            manufacturer="NVIDIA",
            model="Tesla V100",
            description="High-performance server GPU",
        )
        server_gpu.add_constraint(ConstraintType.MEMORY, "max_memory_mb", 32768, "MB")
        server_gpu.add_constraint(
            ConstraintType.COMPUTE, "max_ops_per_sec", 1e14, "ops/sec"
        )
        server_gpu.add_constraint(ConstraintType.POWER, "max_power_watts", 300, "W")
        server_gpu.add_capability("cuda_acceleration", True, version="10.1")
        server_gpu.add_capability("tensor_cores", True, version="V1")
        server_gpu.preferred_quantization = ["float32", "float16", "int8"]

        # Save all default profiles
        for profile in [rpi4, mobile, edge_tpu, server_gpu]:
            self.add_profile(profile)

    def add_profile(self, profile: DeviceProfile) -> None:
        """Add or update a device profile."""
        self.profiles[profile.device_id] = profile
        self._save_profile(profile)
        logger.info(f"Added device profile: {profile.name} ({profile.device_id})")

    def _save_profile(self, profile: DeviceProfile) -> None:
        """Save a profile to disk."""
        profile_file = self.profiles_dir / f"{profile.device_id}.json"

        with open(profile_file, "w") as f:
            json.dump(profile.to_dict(), f, indent=2, default=str)

        # Update cache
        self.profile_cache[profile.device_id] = profile_file.stat().st_mtime

    def get_profile(self, device_id: str) -> Optional[DeviceProfile]:
        """Get a device profile by ID."""
        # Check if profile file has been updated
        profile_file = self.profiles_dir / f"{device_id}.json"
        if profile_file.exists():
            current_mtime = profile_file.stat().st_mtime
            cached_mtime = self.profile_cache.get(device_id, 0)

            if current_mtime > cached_mtime:
                logger.debug(f"Reloading updated profile: {device_id}")
                self._load_profile_file(profile_file)

        return self.profiles.get(device_id)

    def list_profiles(
        self, category: Optional[DeviceCategory] = None
    ) -> List[DeviceProfile]:
        """List all profiles, optionally filtered by category."""
        profiles = list(self.profiles.values())

        if category:
            profiles = [p for p in profiles if p.category == category]

        return sorted(profiles, key=lambda p: p.name)

    def search_profiles(self, query: str) -> List[DeviceProfile]:
        """Search profiles by name, manufacturer, or tags."""
        query = query.lower()
        results = []

        for profile in self.profiles.values():
            if (
                query in profile.name.lower()
                or query in profile.manufacturer.lower()
                or query in profile.description.lower()
                or any(query in tag.lower() for tag in profile.tags)
            ):
                results.append(profile)

        return sorted(results, key=lambda p: p.name)

    def create_profile_from_template(
        self, device_id: str, name: str, template_id: str, **overrides
    ) -> DeviceProfile:
        """Create a new profile based on an existing template."""
        template = self.get_profile(template_id)
        if not template:
            raise ValueError(f"Template profile not found: {template_id}")

        # Create a copy of the template
        profile_data = template.to_dict()
        profile_data.update(
            {
                "device_id": device_id,
                "name": name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "created_by": "user",
            }
        )

        # Apply overrides
        profile_data.update(overrides)

        # Create new profile
        new_profile = DeviceProfile.from_dict(profile_data)
        self.add_profile(new_profile)

        return new_profile

    def update_profile_constraint(
        self,
        device_id: str,
        constraint_name: str,
        new_value: Union[int, float, str],
        **kwargs,
    ) -> bool:
        """Update a specific constraint in a profile."""
        profile = self.get_profile(device_id)
        if not profile:
            return False

        constraint = profile.get_constraint(constraint_name)
        if constraint:
            constraint.value = new_value
            for key, value in kwargs.items():
                if hasattr(constraint, key):
                    setattr(constraint, key, value)
        else:
            # Add new constraint if it doesn't exist
            constraint_type = kwargs.get("constraint_type", ConstraintType.COMPUTE)
            unit = kwargs.get("unit", "")
            profile.add_constraint(
                constraint_type, constraint_name, new_value, unit, **kwargs
            )

        profile.updated_at = datetime.now()
        self._save_profile(profile)
        return True

    def benchmark_device(
        self, device_id: str, benchmark_results: Dict[str, float]
    ) -> None:
        """Update device profile with benchmark results."""
        profile = self.get_profile(device_id)
        if not profile:
            logger.warning(f"Profile not found for benchmarking: {device_id}")
            return

        profile.benchmark_scores.update(benchmark_results)
        profile.updated_at = datetime.now()
        self._save_profile(profile)

        logger.info(f"Updated benchmark scores for {profile.name}")

    def auto_detect_device(self) -> Optional[DeviceProfile]:
        """Attempt to automatically detect the current device."""
        import platform

        import psutil

        # Get system information
        system_info: Dict[str, Union[str, float]] = {
            "platform": str(platform.platform()),
            "processor": str(platform.processor()),
            "architecture": str(platform.architecture()[0]),
            "memory_gb": float(psutil.virtual_memory().total / (1024**3)),
        }

        logger.info(f"Detected system: {system_info}")

        # Try to match against existing profiles
        for profile in self.profiles.values():
            # Simple heuristic matching
            if (
                "raspberry" in str(system_info["platform"]).lower()
                and "raspberry" in profile.name.lower()
            ):
                logger.info(f"Auto-detected device: {profile.name}")
                return profile

        # Create a generic profile for unknown devices
        device_id = f"auto_detected_{int(time.time())}"
        profile = DeviceProfile(
            device_id=device_id,
            name=f"Auto-detected {system_info['platform']}",
            category=DeviceCategory.CUSTOM,
            description=f"Automatically detected device: {system_info['processor']}",
            created_by="auto_detection",
        )

        # Add basic constraints based on detected hardware
        profile.add_constraint(
            ConstraintType.MEMORY,
            "max_memory_mb",
            int(float(system_info["memory_gb"]) * 1024 * 0.8),  # 80% of available
            # memory
            "MB",
            description="Detected system memory (80% allocation)",
        )

        self.add_profile(profile)
        logger.info(f"Created auto-detected profile: {profile.name}")

        return profile

    def export_profiles(self, output_file: str) -> None:
        """Export all profiles to a single JSON file."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "profiles": {
                pid: profile.to_dict() for pid, profile in self.profiles.items()
            },
        }

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported {len(self.profiles)} profiles to {output_file}")

    def import_profiles(self, input_file: str, overwrite: bool = False) -> int:
        """Import profiles from a JSON file."""
        with open(input_file, "r") as f:
            import_data = json.load(f)

        imported_count = 0
        for device_id, profile_data in import_data["profiles"].items():
            if device_id in self.profiles and not overwrite:
                logger.warning(f"Skipping existing profile: {device_id}")
                continue

            try:
                profile = DeviceProfile.from_dict(profile_data)
                self.add_profile(profile)
                imported_count += 1
            except Exception as e:
                logger.error(f"Failed to import profile {device_id}: {e}")

        logger.info(f"Imported {imported_count} profiles from {input_file}")
        return imported_count


# Global profile manager instance
_global_manager: Optional[DeviceProfileManager] = None


def get_profile_manager() -> DeviceProfileManager:
    """Get or create the global device profile manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = DeviceProfileManager()
    return _global_manager


def get_device_profile(device_id: str) -> Optional[DeviceProfile]:
    """Get a device profile by ID."""
    manager = get_profile_manager()
    return manager.get_profile(device_id)


def list_available_devices() -> List[str]:
    """List all available device IDs."""
    manager = get_profile_manager()
    return list(manager.profiles.keys())


if __name__ == "__main__":
    # Example usage
    manager = DeviceProfileManager()

    # List all profiles
    print("Available device profiles:")
    for profile in manager.list_profiles():
        print(f"  - {profile.name} ({profile.device_id}) - {profile.category.value}")

    # Get a specific profile
    rpi_profile = manager.get_profile("raspberry_pi_4")
    if rpi_profile:
        print(f"\nRaspberry Pi 4 Profile:")
        mem_constraint = rpi_profile.get_constraint("max_memory_mb")
        if mem_constraint:
            print(f"  Memory limit: {mem_constraint.value}MB")
        print(f"  Preferred quantization: {rpi_profile.preferred_quantization}")

        # Test model requirements
        requirements = {"max_memory_mb": 2048, "max_ops_per_sec": 1e8}
        violations = rpi_profile.validate_model_requirements(requirements)
        if violations:
            print(f"  Constraint violations: {violations}")
        else:
            print("  ✅ Model requirements satisfied")

    # Auto-detect current device
    detected = manager.auto_detect_device()
    if detected:
        print(f"\nAuto-detected device: {detected.name}")

    print("Device profile management example completed!")
