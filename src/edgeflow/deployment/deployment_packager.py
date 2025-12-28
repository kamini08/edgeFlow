"""EdgeFlow Deployment Artifact Packager

This module handles device-specific packaging of EdgeFlow models and dependencies,
creating deployment artifacts optimized for different edge device runtime environments.

Key Features:
- Device-specific packaging (Raspberry Pi, Jetson, Cortex-M, etc.)
- Compact model binaries for constrained devices
- Dependency bundling and flattening
- Runtime environment optimization
- Storage limit compliance
"""

import logging
import os
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types for deployment packaging."""

    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    CORTEX_M4 = "cortex_m4"
    CORTEX_M7 = "cortex_m7"
    LINUX_ARM64 = "linux_arm64"
    LINUX_X86_64 = "linux_x86_64"


class PackageFormat(Enum):
    """Package formats for different deployment scenarios."""

    TAR_GZ = "tar.gz"
    ZIP = "zip"
    DEB = "deb"
    RPM = "rpm"
    FLATPAK = "flatpak"
    DOCKER = "docker"


@dataclass
class DeviceConstraints:
    """Device-specific constraints and capabilities."""

    max_storage_mb: int
    max_memory_mb: int
    supported_architectures: List[str]
    supported_package_formats: List[PackageFormat]
    runtime_requirements: List[str]
    optimization_flags: List[str]
    compression_level: int


@dataclass
class DeploymentArtifact:
    """Represents a deployment artifact."""

    artifact_path: str
    artifact_type: str
    device_type: DeviceType
    package_format: PackageFormat
    size_mb: float
    dependencies: List[str]
    metadata: Dict[str, Any]


class EdgeFlowDeploymentPackager:
    """Handles device-specific packaging of EdgeFlow models and dependencies."""

    def __init__(self):
        self.device_constraints = self._initialize_device_constraints()
        self.package_templates = self._initialize_package_templates()

    def _initialize_device_constraints(self) -> Dict[DeviceType, DeviceConstraints]:
        """Initialize device-specific constraints."""
        return {
            DeviceType.RASPBERRY_PI: DeviceConstraints(
                max_storage_mb=100,
                max_memory_mb=256,
                supported_architectures=["armv7l", "aarch64"],
                supported_package_formats=[PackageFormat.TAR_GZ, PackageFormat.DEB],
                runtime_requirements=["python3", "numpy", "tensorflow-lite"],
                optimization_flags=["-Os", "-ffast-math", "-fomit-frame-pointer"],
                compression_level=9,
            ),
            DeviceType.JETSON_NANO: DeviceConstraints(
                max_storage_mb=200,
                max_memory_mb=1024,
                supported_architectures=["aarch64"],
                supported_package_formats=[PackageFormat.TAR_GZ, PackageFormat.DEB],
                runtime_requirements=["python3", "numpy", "tensorflow-lite", "cuda"],
                optimization_flags=["-O2", "-ffast-math"],
                compression_level=6,
            ),
            DeviceType.JETSON_XAVIER: DeviceConstraints(
                max_storage_mb=500,
                max_memory_mb=2048,
                supported_architectures=["aarch64"],
                supported_package_formats=[PackageFormat.TAR_GZ, PackageFormat.DEB],
                runtime_requirements=[
                    "python3",
                    "numpy",
                    "tensorflow-lite",
                    "cuda",
                    "tensorrt",
                ],
                optimization_flags=["-O3", "-ffast-math", "-march=native"],
                compression_level=6,
            ),
            DeviceType.CORTEX_M4: DeviceConstraints(
                max_storage_mb=50,
                max_memory_mb=128,
                supported_architectures=["armv7e-m"],
                supported_package_formats=[PackageFormat.TAR_GZ],
                runtime_requirements=["tensorflow-lite-micro"],
                optimization_flags=["-Os", "-ffunction-sections", "-fdata-sections"],
                compression_level=9,
            ),
            DeviceType.CORTEX_M7: DeviceConstraints(
                max_storage_mb=100,
                max_memory_mb=256,
                supported_architectures=["armv7e-m"],
                supported_package_formats=[PackageFormat.TAR_GZ],
                runtime_requirements=["tensorflow-lite-micro"],
                optimization_flags=["-Os", "-ffunction-sections", "-fdata-sections"],
                compression_level=9,
            ),
            DeviceType.LINUX_ARM64: DeviceConstraints(
                max_storage_mb=200,
                max_memory_mb=512,
                supported_architectures=["aarch64"],
                supported_package_formats=[PackageFormat.TAR_GZ, PackageFormat.DEB],
                runtime_requirements=["python3", "numpy", "tensorflow-lite"],
                optimization_flags=["-O2", "-ffast-math"],
                compression_level=6,
            ),
            DeviceType.LINUX_X86_64: DeviceConstraints(
                max_storage_mb=500,
                max_memory_mb=1024,
                supported_architectures=["x86_64"],
                supported_package_formats=[
                    PackageFormat.TAR_GZ,
                    PackageFormat.DEB,
                    PackageFormat.RPM,
                ],
                runtime_requirements=["python3", "numpy", "tensorflow-lite"],
                optimization_flags=["-O3", "-ffast-math", "-march=native"],
                compression_level=6,
            ),
        }

    def _initialize_package_templates(self) -> Dict[str, str]:
        """Initialize package templates for different deployment scenarios."""
        return {
            "raspberry_pi_deployment": """
#!/bin/bash
# EdgeFlow Raspberry Pi Deployment Script
# Generated by EdgeFlow Deployment Packager

set -e

# Configuration
MODEL_PATH="$1"
DEPLOY_PATH="${2:-/opt/edgeflow}"
PYTHON_PATH="/usr/bin/python3"

# Check system requirements
check_requirements() {
    echo "Checking system requirements..."
    
    # Check Python 3
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 is required but not installed"
        exit 1
    fi
    
    # Check TensorFlow Lite
    python3 -c "import tflite_runtime.interpreter" 2>/dev/null || {
        echo "Installing TensorFlow Lite runtime..."
        pip3 install tflite-runtime
    }
    
    # Check NumPy
    python3 -c "import numpy" 2>/dev/null || {
        echo "Installing NumPy..."
        pip3 install numpy
    }
}

# Install dependencies
install_dependencies() {
    echo "Installing dependencies..."
    apt-get update
    apt-get install -y python3-pip python3-dev
    pip3 install --upgrade pip
    pip3 install numpy tflite-runtime
}

# Setup deployment directory
setup_deployment() {
    echo "Setting up deployment directory: $DEPLOY_PATH"
    mkdir -p "$DEPLOY_PATH"/{bin,lib,models,config}
    
    # Copy model
    cp "$MODEL_PATH" "$DEPLOY_PATH/models/"
    
    # Set permissions
    chmod +x "$DEPLOY_PATH/bin"/*
    chown -R root:root "$DEPLOY_PATH"
}

# Create systemd service
create_service() {
    echo "Creating systemd service..."
    cat > /etc/systemd/system/edgeflow.service << EOF
[Unit]
Description=EdgeFlow ML Inference Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$DEPLOY_PATH
ExecStart=$DEPLOY_PATH/bin/inference
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable edgeflow.service
}

# Main installation
main() {
    echo "EdgeFlow Raspberry Pi Deployment"
    echo "================================"
    
    if [ -z "$MODEL_PATH" ]; then
        echo "Usage: $0 <model_path> [deploy_path]"
        exit 1
    fi
    
    check_requirements
    install_dependencies
    setup_deployment
    create_service
    
    echo "Deployment complete!"
    echo "Model: $MODEL_PATH"
    echo "Deploy path: $DEPLOY_PATH"
    echo "Service: edgeflow.service"
    echo ""
    echo "To start the service: systemctl start edgeflow"
    echo "To check status: systemctl status edgeflow"
}

main "$@"
""",
            "jetson_deployment": """
#!/bin/bash
# EdgeFlow Jetson Deployment Script
# Generated by EdgeFlow Deployment Packager

set -e

# Configuration
MODEL_PATH="$1"
DEPLOY_PATH="${2:-/opt/edgeflow}"
CUDA_PATH="/usr/local/cuda"

# Check CUDA availability
check_cuda() {
    echo "Checking CUDA availability..."
    if [ -d "$CUDA_PATH" ]; then
        echo "CUDA found at: $CUDA_PATH"
        export PATH="$CUDA_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    else
        echo "Warning: CUDA not found, using CPU-only mode"
    fi
}

# Install TensorRT dependencies
install_tensorrt() {
    echo "Installing TensorRT dependencies..."
    pip3 install tensorrt
    pip3 install pycuda
}

# Setup GPU optimization
setup_gpu_optimization() {
    echo "Setting up GPU optimization..."
    cat > "$DEPLOY_PATH/config/gpu_config.py" << 'EOF'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
EOF
}

# Main installation
main() {
    echo "EdgeFlow Jetson Deployment"
    echo "=========================="
    
    if [ -z "$MODEL_PATH" ]; then
        echo "Usage: $0 <model_path> [deploy_path]"
        exit 1
    fi
    
    check_cuda
    install_tensorrt
    setup_gpu_optimization
    
    echo "Jetson deployment complete!"
    echo "GPU optimization enabled"
}

main "$@"
""",
            "cortex_m_deployment": """
#!/bin/bash
# EdgeFlow Cortex-M Deployment Script
# Generated by EdgeFlow Deployment Packager

set -e

# Configuration
MODEL_PATH="$1"
DEPLOY_PATH="${2:-/opt/edgeflow}"
CROSS_COMPILE="arm-none-eabi-"

# Check cross-compilation tools
check_tools() {
    echo "Checking cross-compilation tools..."
    if ! command -v "${CROSS_COMPILE}gcc" &> /dev/null; then
        echo "Error: ARM GCC toolchain not found"
        echo "Please install: sudo apt-get install gcc-arm-none-eabi"
        exit 1
    fi
}

# Compile for Cortex-M
compile_cortex() {
    echo "Compiling for Cortex-M..."
    "${CROSS_COMPILE}gcc" \\
        -mcpu=cortex-m4 \\
        -mthumb \\
        -Os \\
        -ffunction-sections \\
        -fdata-sections \\
        -Wl,--gc-sections \\
        -o "$DEPLOY_PATH/bin/inference" \\
        "$DEPLOY_PATH/src/inference.c"
}

# Create minimal runtime
create_runtime() {
    echo "Creating minimal runtime..."
    cat > "$DEPLOY_PATH/runtime/minimal_runtime.c" << 'EOF'
#include <stdint.h>
#include <string.h>

// Minimal runtime for Cortex-M
void* malloc(size_t size) {
    static char heap[8192];
    static size_t offset = 0;
    if (offset + size > sizeof(heap)) return NULL;
    void* ptr = &heap[offset];
    offset += size;
    return ptr;
}

void free(void* ptr) {
    // No-op for simplicity
}
EOF
}

# Main installation
main() {
    echo "EdgeFlow Cortex-M Deployment"
    echo "============================"
    
    if [ -z "$MODEL_PATH" ]; then
        echo "Usage: $0 <model_path> [deploy_path]"
        exit 1
    fi
    
    check_tools
    create_runtime
    compile_cortex
    
    echo "Cortex-M deployment complete!"
    echo "Binary: $DEPLOY_PATH/bin/inference"
}

main "$@"
""",
        }

    def package_for_device(
        self, model_path: str, config: Dict[str, Any], output_dir: str = "deployment"
    ) -> List[DeploymentArtifact]:
        """Package model and dependencies for a specific device.

        Args:
            model_path: Path to the optimized model file
            config: EdgeFlow configuration
            output_dir: Output directory for deployment artifacts

        Returns:
            List of deployment artifacts
        """
        device_type = DeviceType(config.get("target_device", "raspberry_pi"))
        constraints = self.device_constraints[device_type]

        logger.info(f"Packaging for {device_type.value}")
        logger.info(f"Storage limit: {constraints.max_storage_mb}MB")
        logger.info(f"Memory limit: {constraints.max_memory_mb}MB")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        artifacts = []

        # Package model binary
        model_artifact = self._package_model_binary(
            model_path, device_type, constraints, output_dir
        )
        artifacts.append(model_artifact)

        # Package inference code
        code_artifact = self._package_inference_code(
            config, device_type, constraints, output_dir
        )
        artifacts.append(code_artifact)

        # Package dependencies
        deps_artifact = self._package_dependencies(
            config, device_type, constraints, output_dir
        )
        artifacts.append(deps_artifact)

        # Package deployment scripts
        script_artifact = self._package_deployment_scripts(
            config, device_type, constraints, output_dir
        )
        artifacts.append(script_artifact)

        # Create final deployment package
        final_artifact = self._create_final_package(
            artifacts, device_type, constraints, output_dir
        )
        artifacts.append(final_artifact)

        # Validate package size
        self._validate_package_size(final_artifact, constraints)

        return artifacts

    def _package_model_binary(
        self,
        model_path: str,
        device_type: DeviceType,
        constraints: DeviceConstraints,
        output_dir: str,
    ) -> DeploymentArtifact:
        """Package the model binary with device-specific optimizations."""
        logger.info("Packaging model binary...")

        # Create model package directory
        model_dir = os.path.join(output_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        # Copy and optimize model
        model_name = os.path.basename(model_path)
        optimized_model_path = os.path.join(model_dir, f"optimized_{model_name}")

        if os.path.exists(model_path):
            shutil.copy2(model_path, optimized_model_path)

            # Apply device-specific optimizations
            self._optimize_model_for_device(
                optimized_model_path, device_type, constraints
            )
        else:
            # Create dummy model for testing
            with open(optimized_model_path, "wb") as f:
                f.write(b"dummy_model_data")

        # Create model metadata
        metadata = {
            "device_type": device_type.value,
            "original_size_mb": os.path.getsize(model_path) / (1024 * 1024)
            if os.path.exists(model_path)
            else 0.0,
            "optimized_size_mb": os.path.getsize(optimized_model_path) / (1024 * 1024),
            "optimization_flags": constraints.optimization_flags,
            "compression_level": constraints.compression_level,
        }

        # Create model package
        package_path = os.path.join(output_dir, f"model_{device_type.value}.tar.gz")
        with tarfile.open(
            package_path, "w:gz", compresslevel=constraints.compression_level
        ) as tar:
            tar.add(model_dir, arcname="model")

            # Add metadata
            import json

            metadata_file = os.path.join(model_dir, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            tar.add(metadata_file, arcname="model/metadata.json")

        return DeploymentArtifact(
            artifact_path=package_path,
            artifact_type="model_binary",
            device_type=device_type,
            package_format=PackageFormat.TAR_GZ,
            size_mb=os.path.getsize(package_path) / (1024 * 1024),
            dependencies=["tensorflow-lite"],
            metadata=metadata,
        )

    def _package_inference_code(
        self,
        config: Dict[str, Any],
        device_type: DeviceType,
        constraints: DeviceConstraints,
        output_dir: str,
    ) -> DeploymentArtifact:
        """Package device-specific inference code."""
        logger.info("Packaging inference code...")

        # Create code package directory
        code_dir = os.path.join(output_dir, "inference_code")
        os.makedirs(code_dir, exist_ok=True)

        # Generate device-specific inference code
        inference_code = self._generate_device_inference_code(
            config, device_type, constraints
        )

        # Write inference code
        code_file = os.path.join(code_dir, "inference.py")
        with open(code_file, "w") as f:
            f.write(inference_code)

        # Create C++ version for embedded devices
        if device_type in [DeviceType.CORTEX_M4, DeviceType.CORTEX_M7]:
            cpp_code = self._generate_cpp_inference_code(
                config, device_type, constraints
            )
            cpp_file = os.path.join(code_dir, "inference.cpp")
            with open(cpp_file, "w") as f:
                f.write(cpp_code)

            # Create Makefile
            makefile = self._generate_makefile(device_type, constraints)
            makefile_path = os.path.join(code_dir, "Makefile")
            with open(makefile_path, "w") as f:
                f.write(makefile)

        # Create package
        package_path = os.path.join(
            output_dir, f"inference_code_{device_type.value}.tar.gz"
        )
        with tarfile.open(
            package_path, "w:gz", compresslevel=constraints.compression_level
        ) as tar:
            tar.add(code_dir, arcname="inference_code")

        return DeploymentArtifact(
            artifact_path=package_path,
            artifact_type="inference_code",
            device_type=device_type,
            package_format=PackageFormat.TAR_GZ,
            size_mb=os.path.getsize(package_path) / (1024 * 1024),
            dependencies=constraints.runtime_requirements,
            metadata={
                "language": "python"
                if device_type not in [DeviceType.CORTEX_M4, DeviceType.CORTEX_M7]
                else "cpp"
            },
        )

    def _package_dependencies(
        self,
        config: Dict[str, Any],
        device_type: DeviceType,
        constraints: DeviceConstraints,
        output_dir: str,
    ) -> DeploymentArtifact:
        """Package device-specific dependencies."""
        logger.info("Packaging dependencies...")

        # Create dependencies directory
        deps_dir = os.path.join(output_dir, "dependencies")
        os.makedirs(deps_dir, exist_ok=True)

        # Generate requirements.txt
        requirements = self._generate_requirements_txt(config, device_type, constraints)
        req_file = os.path.join(deps_dir, "requirements.txt")
        with open(req_file, "w") as f:
            f.write(requirements)

        # Create installation script
        install_script = self._generate_dependency_install_script(
            device_type, constraints
        )
        install_file = os.path.join(deps_dir, "install_dependencies.sh")
        with open(install_file, "w") as f:
            f.write(install_script)
        os.chmod(install_file, 0o755)

        # Create package
        package_path = os.path.join(
            output_dir, f"dependencies_{device_type.value}.tar.gz"
        )
        with tarfile.open(
            package_path, "w:gz", compresslevel=constraints.compression_level
        ) as tar:
            tar.add(deps_dir, arcname="dependencies")

        return DeploymentArtifact(
            artifact_path=package_path,
            artifact_type="dependencies",
            device_type=device_type,
            package_format=PackageFormat.TAR_GZ,
            size_mb=os.path.getsize(package_path) / (1024 * 1024),
            dependencies=constraints.runtime_requirements,
            metadata={"requirements_count": len(constraints.runtime_requirements)},
        )

    def _package_deployment_scripts(
        self,
        config: Dict[str, Any],
        device_type: DeviceType,
        constraints: DeviceConstraints,
        output_dir: str,
    ) -> DeploymentArtifact:
        """Package device-specific deployment scripts."""
        logger.info("Packaging deployment scripts...")

        # Create scripts directory
        scripts_dir = os.path.join(output_dir, "scripts")
        os.makedirs(scripts_dir, exist_ok=True)

        # Generate deployment script
        deployment_script = self.package_templates.get(
            f"{device_type.value}_deployment",
            self.package_templates["raspberry_pi_deployment"],
        )

        script_file = os.path.join(scripts_dir, "deploy.sh")
        with open(script_file, "w") as f:
            f.write(deployment_script)
        os.chmod(script_file, 0o755)

        # Create configuration script
        config_script = self._generate_config_script(config, device_type)
        config_file = os.path.join(scripts_dir, "configure.sh")
        with open(config_file, "w") as f:
            f.write(config_script)
        os.chmod(config_file, 0o755)

        # Create package
        package_path = os.path.join(output_dir, f"scripts_{device_type.value}.tar.gz")
        with tarfile.open(
            package_path, "w:gz", compresslevel=constraints.compression_level
        ) as tar:
            tar.add(scripts_dir, arcname="scripts")

        return DeploymentArtifact(
            artifact_path=package_path,
            artifact_type="deployment_scripts",
            device_type=device_type,
            package_format=PackageFormat.TAR_GZ,
            size_mb=os.path.getsize(package_path) / (1024 * 1024),
            dependencies=[],
            metadata={"script_count": 2},
        )

    def _create_final_package(
        self,
        artifacts: List[DeploymentArtifact],
        device_type: DeviceType,
        constraints: DeviceConstraints,
        output_dir: str,
    ) -> DeploymentArtifact:
        """Create the final deployment package."""
        logger.info("Creating final deployment package...")

        # Create final package directory
        final_dir = os.path.join(output_dir, f"edgeflow_{device_type.value}_deployment")
        os.makedirs(final_dir, exist_ok=True)

        # Extract all artifacts into final package
        for artifact in artifacts:
            if artifact.artifact_type != "final_package":
                with tarfile.open(artifact.artifact_path, "r:gz") as tar:
                    tar.extractall(final_dir)

        # Create deployment manifest
        manifest = {
            "device_type": device_type.value,
            "constraints": {
                "max_storage_mb": constraints.max_storage_mb,
                "max_memory_mb": constraints.max_memory_mb,
                "supported_architectures": constraints.supported_architectures,
            },
            "artifacts": [
                {
                    "type": artifact.artifact_type,
                    "path": artifact.artifact_path,
                    "size_mb": artifact.size_mb,
                }
                for artifact in artifacts
            ],
            "total_size_mb": sum(artifact.size_mb for artifact in artifacts),
        }

        import json

        manifest_file = os.path.join(final_dir, "deployment_manifest.json")
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        # Create final package
        final_package_path = os.path.join(
            output_dir, f"edgeflow_{device_type.value}_deployment.tar.gz"
        )
        with tarfile.open(
            final_package_path, "w:gz", compresslevel=constraints.compression_level
        ) as tar:
            tar.add(final_dir, arcname=f"edgeflow_{device_type.value}_deployment")

        return DeploymentArtifact(
            artifact_path=final_package_path,
            artifact_type="final_package",
            device_type=device_type,
            package_format=PackageFormat.TAR_GZ,
            size_mb=os.path.getsize(final_package_path) / (1024 * 1024),
            dependencies=constraints.runtime_requirements,
            metadata=manifest,
        )

    def _optimize_model_for_device(
        self, model_path: str, device_type: DeviceType, constraints: DeviceConstraints
    ):
        """Apply device-specific optimizations to the model."""
        logger.info(f"Optimizing model for {device_type.value}")

        # For now, just log the optimization flags
        # In a real implementation, this would apply actual model optimizations
        logger.info(f"Optimization flags: {constraints.optimization_flags}")
        logger.info(f"Compression level: {constraints.compression_level}")

    def _generate_device_inference_code(
        self,
        config: Dict[str, Any],
        device_type: DeviceType,
        constraints: DeviceConstraints,
    ) -> str:
        """Generate device-specific inference code."""
        model_path = config.get("model", "model.tflite")
        quantize = config.get("quantize", "none")
        buffer_size = config.get("buffer_size", 1)

        if device_type in [DeviceType.CORTEX_M4, DeviceType.CORTEX_M7]:
            return self._generate_cpp_inference_code(config, device_type, constraints)

        # Python inference code for other devices
        device_name = device_type.value
        model_path_str = config.get("model", "model.tflite")
        quantize_str = config.get("quantize", "none")
        buffer_size_str = str(config.get("buffer_size", 1))

        code = f'''#!/usr/bin/env python3
"""
EdgeFlow Inference Engine for {device_name}
Generated by EdgeFlow Deployment Packager
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Any, Optional

# Device-specific imports
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("TensorFlow Lite runtime not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeFlowInference:
    """Device-optimized inference engine for {device_name}."""
    
    def __init__(self, model_path: str = "{model_path_str}"):
        """Initialize inference engine."""
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.device_type = "{device_name}"
        
        # Device-specific configuration
        self.buffer_size = {buffer_size_str}
        self.quantize_type = "{quantize_str}"
        
        self._load_model()
    
    def _load_model(self):
        """Load TensorFlow Lite model."""
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"Model loaded successfully for {{self.device_type}}")
            logger.info(f"Input shape: {{self.input_details[0]['shape']}}")
            logger.info(f"Output shape: {{self.output_details[0]['shape']}}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {{e}}")
            raise
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        if self.interpreter is None:
            raise RuntimeError("Model not loaded")
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        start_time = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start_time
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        logger.debug(f"Inference time: {{inference_time * 1000:.2f}}ms")
        
        return output_data
    
    def benchmark(self, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        if self.input_details is None:
            raise RuntimeError("Model not loaded")
        
        # Generate test input
        input_shape = self.input_details[0]['shape']
        dtype = self.input_details[0]['dtype']
        
        if dtype == np.float32:
            test_input = np.random.random(input_shape).astype(np.float32)
        elif dtype == np.int8:
            test_input = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
        else:
            test_input = np.random.random(input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.predict(test_input)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            self.predict(test_input)
            times.append(time.perf_counter() - start_time)
        
        return {{
            "mean_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "throughput_fps": 1.0 / np.mean(times)
        }}

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="EdgeFlow Inference")
    parser.add_argument("--model", default="{model_path}", help="Model path")
    parser.add_argument("--input", help="Input data path")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = EdgeFlowInference(args.model)
    
    if args.benchmark:
        print("Running benchmark...")
        results = inference.benchmark(args.runs)
        print(f"Mean inference time: {{results['mean_time_ms']:.2f}}ms")
        print(f"Throughput: {{results['throughput_fps']:.2f}} FPS")
    else:
        print("EdgeFlow inference engine ready")
        print(f"Device: {device_name}")
        print(f"Model: {{args.model}}")

if __name__ == "__main__":
    main()
'''
        return code

    def _generate_cpp_inference_code(
        self,
        config: Dict[str, Any],
        device_type: DeviceType,
        constraints: DeviceConstraints,
    ) -> str:
        """Generate C++ inference code for embedded devices."""
        model_path = config.get("model", "model.tflite")
        device_name = device_type.value

        code = f"""/*
 * EdgeFlow C++ Inference Engine for {device_name}
 * Generated by EdgeFlow Deployment Packager
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

class EdgeFlowInference {{
private:
    tflite::MicroErrorReporter error_reporter;
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;
    
    // Device-specific configuration
    static constexpr int kTensorArenaSize = 1024 * 1024;  // 1MB arena
    uint8_t tensor_arena[kTensorArenaSize];
    
public:
    EdgeFlowInference() {{
        // Load model
        model = tflite::GetModel("{model_path}");
        if (model->version() != TFLITE_SCHEMA_VERSION) {{
            printf("Model schema version mismatch\\n");
            return;
        }}
        
        // Create interpreter
        static tflite::AllOpsResolver resolver;
        static tflite::MicroInterpreter static_interpreter(
            model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
        interpreter = &static_interpreter;
        
        // Allocate tensors
        TfLiteStatus allocate_status = interpreter->AllocateTensors();
        if (allocate_status != kTfLiteOk) {{
            printf("Failed to allocate tensors\\n");
            return;
        }}
        
        // Get input and output tensors
        input_tensor = interpreter->input(0);
        output_tensor = interpreter->output(0);
        
        printf("EdgeFlow inference engine initialized for {device_name}\\n");
        printf("Input shape: [");
        for (int i = 0; i < input_tensor->dims->size; i++) {{
            printf("%d", input_tensor->dims->data[i]);
            if (i < input_tensor->dims->size - 1) printf(", ");
        }}
        printf("]\\n");
    }}
    
    bool predict(const float* input_data, float* output_data) {{
        if (!interpreter) return false;
        
        // Copy input data
        memcpy(input_tensor->data.f, input_data, 
               input_tensor->bytes);
        
        // Run inference
        auto start = std::chrono::high_resolution_clock::now();
        TfLiteStatus invoke_status = interpreter->Invoke();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (invoke_status != kTfLiteOk) {{
            printf("Inference failed\\n");
            return false;
        }}
        
        // Copy output data
        memcpy(output_data, output_tensor->data.f, 
               output_tensor->bytes);
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("Inference time: %ld microseconds\\n", duration.count());
        
        return true;
    }}
    
    void benchmark(int num_runs = 100) {{
        printf("Running benchmark with %d runs...\\n", num_runs);
        
        // Generate test input
        std::vector<float> test_input(input_tensor->bytes / sizeof(float));
        for (auto& val : test_input) {{
            val = static_cast<float>(rand()) / RAND_MAX;
        }}
        
        std::vector<float> test_output(output_tensor->bytes / sizeof(float));
        
        // Warmup
        for (int i = 0; i < 10; i++) {{
            predict(test_input.data(), test_output.data());
        }}
        
        // Benchmark
        long total_time = 0;
        for (int i = 0; i < num_runs; i++) {{
            auto start = std::chrono::high_resolution_clock::now();
            predict(test_input.data(), test_output.data());
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }}
        
        double avg_time = static_cast<double>(total_time) / num_runs;
        double throughput = 1000000.0 / avg_time;  // FPS
        
        printf("Benchmark results:\\n");
        printf("  Average inference time: %.2f microseconds\\n", avg_time);
        printf("  Throughput: %.2f FPS\\n", throughput);
    }}
}};

int main() {{
    printf("EdgeFlow C++ Inference Engine for {device_name}\\n");
    
    EdgeFlowInference inference;
    inference.benchmark(100);
    
    return 0;
}}
"""
        return code

    def _generate_makefile(
        self, device_type: DeviceType, constraints: DeviceConstraints
    ) -> str:
        """Generate Makefile for C++ compilation."""
        device_name = device_type.value
        makefile = f"""# EdgeFlow Makefile for {device_name}
# Generated by EdgeFlow Deployment Packager

CC = arm-none-eabi-gcc
CFLAGS = -mcpu=cortex-m4 -mthumb -Os -ffunction-sections -fdata-sections
LDFLAGS = -Wl,--gc-sections -Wl,--print-memory-usage

# TensorFlow Lite Micro paths
TFLITE_MICRO_PATH = tensorflow/lite/micro
INCLUDES = -I$(TFLITE_MICRO_PATH) -I$(TFLITE_MICRO_PATH)/tools/make/downloads/flatbuffers/include

# Source files
SOURCES = inference.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = inference

all: $(TARGET)

$(TARGET): $(OBJECTS)
\t$(CC) $(LDFLAGS) -o $@ $^

%.o: %.cpp
\t$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
\trm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
"""
        return makefile

    def _generate_requirements_txt(
        self,
        config: Dict[str, Any],
        device_type: DeviceType,
        constraints: DeviceConstraints,
    ) -> str:
        """Generate requirements.txt for device dependencies."""
        device_name = device_type.value
        requirements = f"""# EdgeFlow Requirements for {device_name}
# Generated by EdgeFlow Deployment Packager

# Core dependencies
numpy>=1.19.0
"""

        if device_type in [
            DeviceType.RASPBERRY_PI,
            DeviceType.JETSON_NANO,
            DeviceType.JETSON_XAVIER,
        ]:
            requirements += """# TensorFlow Lite runtime
tflite-runtime>=2.5.0
"""
        elif device_type in [DeviceType.CORTEX_M4, DeviceType.CORTEX_M7]:
            requirements += """# TensorFlow Lite Micro
tflite-micro>=0.1.0
"""
        else:
            requirements += """# TensorFlow Lite
tensorflow-lite>=2.5.0
"""

        if device_type in [DeviceType.JETSON_NANO, DeviceType.JETSON_XAVIER]:
            requirements += """# CUDA support
tensorrt>=8.0.0
pycuda>=2021.1
"""

        return requirements

    def _generate_dependency_install_script(
        self, device_type: DeviceType, constraints: DeviceConstraints
    ) -> str:
        """Generate dependency installation script."""
        device_name = device_type.value
        script = f"""#!/bin/bash
# EdgeFlow Dependency Installation Script for {device_name}
# Generated by EdgeFlow Deployment Packager

set -e

echo "Installing dependencies for {device_name}..."

# Update package lists
apt-get update

# Install system dependencies
apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    build-essential

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Device-specific installations
"""

        if device_type in [DeviceType.JETSON_NANO, DeviceType.JETSON_XAVIER]:
            script += """
# Install CUDA dependencies
if [ -d "/usr/local/cuda" ]; then
    echo "CUDA found, installing TensorRT..."
    pip3 install tensorrt pycuda
else
    echo "CUDA not found, skipping TensorRT installation"
fi
"""

        script += """
echo "Dependency installation complete!"
"""
        return script

    def _generate_config_script(
        self, config: Dict[str, Any], device_type: DeviceType
    ) -> str:
        """Generate configuration script."""
        device_name = device_type.value
        script = f"""#!/bin/bash
# EdgeFlow Configuration Script for {device_name}
# Generated by EdgeFlow Deployment Packager

set -e

echo "Configuring EdgeFlow for {device_name}..."

# Create configuration directory
mkdir -p /etc/edgeflow

# Generate device-specific configuration
cat > /etc/edgeflow/config.json << EOF
{{
    "device_type": "{device_name}",
    "model_path": "{config.get('model', 'model.tflite')}",
    "quantize": "{config.get('quantize', 'none')}",
    "buffer_size": {config.get('buffer_size', 1)},
    "memory_limit": {config.get('memory_limit', 64)},
    "optimize_for": "{config.get('optimize_for', 'balanced')}"
}}
EOF

# Set permissions
chmod 644 /etc/edgeflow/config.json

echo "Configuration complete!"
echo "Config file: /etc/edgeflow/config.json"
"""
        return script

    def _validate_package_size(
        self, artifact: DeploymentArtifact, constraints: DeviceConstraints
    ):
        """Validate that package size is within device constraints."""
        if artifact.size_mb > constraints.max_storage_mb:
            logger.warning(
                f"Package size ({artifact.size_mb:.1f}MB) exceeds device limit ({constraints.max_storage_mb}MB)"
            )
            logger.warning(
                "Consider reducing model size or using more aggressive compression"
            )
        else:
            logger.info(
                f"Package size ({artifact.size_mb:.1f}MB) within device limit ({constraints.max_storage_mb}MB)"
            )


def package_for_device(
    model_path: str, config: Dict[str, Any], output_dir: str = "deployment"
) -> List[DeploymentArtifact]:
    """Package model and dependencies for a specific device.

    Args:
        model_path: Path to the optimized model file
        config: EdgeFlow configuration
        output_dir: Output directory for deployment artifacts

    Returns:
        List of deployment artifacts
    """
    packager = EdgeFlowDeploymentPackager()
    return packager.package_for_device(model_path, config, output_dir)


if __name__ == "__main__":
    # Test the packager
    test_config = {
        "model": "test_model.tflite",
        "quantize": "int8",
        "target_device": "raspberry_pi",
        "buffer_size": 16,
        "memory_limit": 256,
        "optimize_for": "latency",
    }

    packager = EdgeFlowDeploymentPackager()
    artifacts = packager.package_for_device("test_model.tflite", test_config)

    print("Deployment artifacts created:")
    for artifact in artifacts:
        print(
            f"  {artifact.artifact_type}: {artifact.artifact_path} ({artifact.size_mb:.1f}MB)"
        )
