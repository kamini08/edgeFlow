"""Code Generator for EdgeFlow DSL.

This module generates target-specific inference code from EdgeFlow AST and IR.
It supports multiple backends:
- C/C++ for bare-metal or embedded Linux environments
- Wrappers for ONNX Runtime, TensorRT, or TVM acceleration
- Glue code for I/O handling (camera, display, network streaming)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set

from edgeflow_ast import (
    ASTVisitor,
    BinaryExpression,
    BufferSizeStatement,
    Condition,
    ConditionalStatement,
    DeployPathStatement,
    FineTuningStatement,
    FrameworkStatement,
    FusionStatement,
    HybridOptimizationStatement,
    Identifier,
    InputStreamStatement,
    LayerDeclaration,
    Literal,
    MemoryLimitStatement,
    ModelStatement,
    OptimizeForStatement,
    PipelineStatement,
    Program,
    PyTorchQuantizeStatement,
    QuantizeStatement,
    TargetDeviceStatement,
    UnaryExpression,
)

# Import IR classes for advanced code generation
try:
    from edgeflow_ir import IRGraph, NodeType

    IR_AVAILABLE = True
except ImportError:
    IR_AVAILABLE = False


class CodeGenerator(ASTVisitor):
    """Generates target-specific inference code from EdgeFlow AST and IR."""

    def __init__(self, ast: Program, ir_graph: Optional[IRGraph] = None):
        self.ast = ast
        self.ir_graph = ir_graph
        self.config: Dict[str, Any] = {}
        self.imports: Set[str] = set()
        self.optimizations: List[str] = []
        self.target_backend: str = "cpp"  # Default to C++

    def generate_python_inference(self) -> str:
        """Generate Python inference code."""
        self.imports.clear()
        self.optimizations.clear()

        # Collect configuration from AST
        self.ast.accept(self)

        # Generate Python code
        return self._generate_python_code()

    def generate_cpp_inference(self) -> str:
        """Generate C++ inference code."""
        self.imports.clear()
        self.optimizations.clear()

        # Collect configuration from AST
        self.ast.accept(self)

        # Generate C++ code
        return self._generate_cpp_code()

    def generate_ir_based_code(self, target_backend: str = "cpp") -> str:
        """Generate target-specific code from IR graph."""
        if not IR_AVAILABLE or not self.ir_graph:
            return self.generate_cpp_inference()  # Fallback to AST-based generation

        self.target_backend = target_backend

        if target_backend == "cpp":
            return self._generate_cpp_from_ir()
        elif target_backend == "onnx":
            return self._generate_onnx_wrapper()
        elif target_backend == "tensorrt":
            return self._generate_tensorrt_wrapper()
        elif target_backend == "tvm":
            return self._generate_tvm_wrapper()
        else:
            return self._generate_cpp_from_ir()

    def _generate_cpp_from_ir(self) -> str:
        """Generate C++ code from IR graph for bare-metal/embedded Linux."""
        code = self._generate_cpp_header()
        code += self._generate_cpp_includes()
        code += self._generate_cpp_io_handling()
        code += self._generate_cpp_inference_engine()
        code += self._generate_cpp_main()
        return code

    def _generate_onnx_wrapper(self) -> str:
        """Generate ONNX Runtime wrapper code."""
        code = self._generate_python_header()
        code += "import onnxruntime as ort\n"
        code += "import numpy as np\n\n"
        code += self._generate_onnx_inference_class()
        return code

    def _generate_tensorrt_wrapper(self) -> str:
        """Generate TensorRT wrapper code."""
        code = self._generate_python_header()
        code += "import tensorrt as trt\n"
        code += "import numpy as np\n\n"
        code += self._generate_tensorrt_inference_class()
        return code

    def _generate_tvm_wrapper(self) -> str:
        """Generate TVM wrapper code."""
        code = self._generate_python_header()
        code += "import tvm\n"
        code += "import numpy as np\n\n"
        code += self._generate_tvm_inference_class()
        return code

    def generate_optimization_report(self) -> str:
        """Generate a report of applied optimizations."""
        self.ast.accept(self)
        return self._generate_optimization_report()

    # AST Visitor Methods
    def visit_program(self, node: Program) -> Any:
        for stmt in node.statements:
            stmt.accept(self)
        return None

    def visit_model_statement(self, node: ModelStatement) -> Any:
        self.config["model_path"] = node.path
        return None

    def visit_quantize_statement(self, node: QuantizeStatement) -> Any:
        self.config["quantization"] = node.quant_type
        if node.quant_type != "none":
            self.optimizations.append(f"Applied {node.quant_type.upper()} quantization")
        return None

    def visit_target_device_statement(self, node: TargetDeviceStatement) -> Any:
        self.config["target_device"] = node.device
        return None

    def visit_deploy_path_statement(self, node: DeployPathStatement) -> Any:
        self.config["deploy_path"] = node.path
        return None

    def visit_input_stream_statement(self, node: InputStreamStatement) -> Any:
        self.config["input_stream"] = node.stream
        return None

    def visit_buffer_size_statement(self, node: BufferSizeStatement) -> Any:
        self.config["buffer_size"] = node.size
        self.optimizations.append(
            f"Set buffer size to {node.size} for optimal streaming"
        )
        return None

    def visit_optimize_for_statement(self, node: OptimizeForStatement) -> Any:
        self.config["optimize_for"] = node.goal
        self.optimizations.append(f"Optimized for {node.goal}")
        return None

    def visit_memory_limit_statement(self, node: MemoryLimitStatement) -> Any:
        self.config["memory_limit"] = node.limit_mb.value
        self.optimizations.append(f"Memory limit set to {node.limit_mb}MB")
        return None

    def visit_fusion_statement(self, node: FusionStatement) -> Any:
        self.config["enable_fusion"] = node.enabled
        if node.enabled:
            self.optimizations.append("Enabled operation fusion")
        return None

    def visit_framework_statement(self, node: FrameworkStatement) -> Any:
        self.config["framework"] = node.framework
        self.optimizations.append(f"Framework set to {node.framework}")
        return None

    def visit_hybrid_optimization_statement(self, node: HybridOptimizationStatement) -> Any:
        self.config["enable_hybrid_optimization"] = node.enabled
        if node.enabled:
            self.optimizations.append("Enabled hybrid optimization pipeline")
        return None

    def visit_pytorch_quantize_statement(self, node: PyTorchQuantizeStatement) -> Any:
        self.config["pytorch_quantize"] = node.quantize_type
        if node.quantize_type != "none":
            self.optimizations.append(f"Applied PyTorch {node.quantize_type} quantization")
        return None

    def visit_fine_tuning_statement(self, node: FineTuningStatement) -> Any:
        self.config["fine_tuning"] = node.enabled
        if node.enabled:
            self.optimizations.append("Enabled fine-tuning capabilities")
        return None

    def visit_conditional_statement(self, node: ConditionalStatement) -> Any:
        # Handle conditional logic
        return None

    def visit_pipeline_statement(self, node: PipelineStatement) -> Any:
        self.config["pipeline_steps"] = node.steps
        return None

    def visit_layer_declaration(self, node: LayerDeclaration) -> Any:
        """Visit a layer declaration node."""
        # Extract layer information for code generation
        layer_info = {
            "name": node.name,
            "type": node.layer_type.value,
            "parameters": node.parameters,
        }
        
        # Add to optimizations list for reporting
        self.optimizations.append(f"Layer {node.name}: {node.layer_type.value}")
        
        # Store layer information in config for code generation
        if "layers" not in self.config:
            self.config["layers"] = []
        self.config["layers"].append(layer_info)
        
        return None

    def visit_literal(self, node: Literal) -> Any:
        return node.value

    def visit_identifier(self, node: Identifier) -> Any:
        return node.name

    def visit_binary_expression(self, node: BinaryExpression) -> Any:
        left = node.left.accept(self)
        right = node.right.accept(self)
        return f"{left} {node.operator} {right}"

    def visit_unary_expression(self, node: UnaryExpression) -> Any:
        operand = node.operand.accept(self)
        return f"{node.operator}{operand}"

    def visit_condition(self, node: Condition) -> Any:
        left = node.left.accept(self)
        right = node.right.accept(self)
        return f"{left} {node.operator} {right}"

    def _generate_python_code(self) -> str:
        """Generate Python inference code."""
        # Add required imports
        self.imports.update(
            [
                "import tensorflow as tf",
                "import numpy as np",
                "import cv2",
                "import time",
                "from typing import Optional, Union, List, Tuple",
            ]
        )

        # Add device-specific imports
        if self.config.get("target_device") == "raspberry_pi":
            self.imports.add("import picamera")

        # Generate the Python class
        code = self._generate_python_header()
        code += self._generate_python_class()
        code += self._generate_python_main()

        return code

    def _generate_python_header(self) -> str:
        """Generate Python file header with imports and docstring."""
        header = '"""EdgeFlow Generated Python Inference Code.\n\n'
        header += "This file was automatically generated by EdgeFlow DSL compiler.\n"
        header += "Do not edit manually - changes will be overwritten.\n\n"
        header += "Configuration:\n"
        for key, value in self.config.items():
            header += f"  {key}: {value}\n"
        header += '"""\n\n'

        # Add imports
        for imp in sorted(self.imports):
            header += imp + "\n"
        header += "\n"

        return header

    def _generate_python_class(self) -> str:
        """Generate the main Python inference class."""
        class_name = "EdgeFlowInference"
        model_path = self.config.get("model_path", "model.tflite")
        quant_type = self.config.get("quantization", "none")
        buffer_size = self.config.get("buffer_size", 32)
        memory_limit = self.config.get("memory_limit", 64)
        optimize_for = self.config.get("optimize_for", "latency")

        code = f"class {class_name}:\n"
        code += '    """EdgeFlow inference engine for edge devices."""\n\n'

        # Constructor
        code += "    def __init__(self, model_path: str = None):\n"
        code += '        """Initialize the inference engine."""\n'
        code += f'        self.model_path = model_path or "{model_path}"\n'
        code += f"        self.buffer_size = {buffer_size}\n"
        code += (
            f"        self.memory_limit = {memory_limit} * 1024 * 1024\n"  # MB to bytes
        )
        code += f'        self.optimize_for = "{optimize_for}"\n'
        code += "        self.interpreter = None\n"
        code += "        self.input_details = None\n"
        code += "        self.output_details = None\n"
        code += "        self._setup_memory_management()\n"
        code += "        self._load_model()\n\n"

        # Memory management
        code += self._generate_python_memory_management()

        # Model loading
        code += self._generate_python_model_loading(quant_type)

        # Input processing
        code += self._generate_python_input_processing()

        # Inference methods
        code += self._generate_python_inference_methods()

        # Utility methods
        code += self._generate_python_utility_methods()

        return code

    def _generate_python_memory_management(self) -> str:
        """Generate memory management code."""
        code = "    def _setup_memory_management(self):\n"
        code += '        """Setup memory management for edge device."""\n'
        code += "        import gc\n"
        code += (
            "        gc.set_threshold(100, 10, 10)  # Aggressive garbage collection\n"
        )
        code += "        \n"
        code += "        # Set TensorFlow memory growth\n"
        code += "        gpus = tf.config.experimental.list_physical_devices('GPU')\n"
        code += "        if gpus:\n"
        code += "            try:\n"
        code += "                for gpu in gpus:\n"
        code += (
            "                    tf.config.experimental.set_memory_growth(gpu, True)\n"
        )
        code += "            except RuntimeError as e:\n"
        code += '                print(f"GPU memory growth setup failed: {e}")\n\n'
        return code

    def _generate_python_model_loading(self, quant_type: str) -> str:
        """Generate model loading code."""
        code = "    def _load_model(self):\n"
        code += '        """Load and configure the TensorFlow Lite model."""\n'
        code += "        try:\n"
        code += "            # Load TFLite model\n"
        code += "            self.interpreter = tf.lite.Interpreter(\n"
        code += "                model_path=self.model_path,\n"
        code += "                experimental_preserve_all_tensors=True\n"
        code += "            )\n"
        code += "            self.interpreter.allocate_tensors()\n"
        code += "            \n"
        code += "            # Get input and output details\n"
        code += (
            "            self.input_details = self.interpreter.get_input_details()\n"
        )
        code += (
            "            self.output_details = self.interpreter.get_output_details()\n"
        )
        code += "            \n"
        code += '            print(f"Model loaded: {self.model_path}")\n'
        code += (
            "            print(f\"Input shape: {self.input_details[0]['shape']}\")\n"
        )
        code += (
            "            print(f\"Output shape: {self.output_details[0]['shape']}\")\n"
        )

        if quant_type != "none":
            code += f'            print(f"Quantization: {quant_type.upper()}")\n'

        code += "        except Exception as e:\n"
        code += '            raise RuntimeError(f"Failed to load model: {e}")\n\n'
        return code

    def _generate_python_input_processing(self) -> str:
        """Generate input processing code."""
        input_stream = self.config.get("input_stream", "camera")

        code = (
            "    def _preprocess_input(self, input_data: "
            "Union[np.ndarray, str]) -> np.ndarray:\n"
        )
        code += '        """Preprocess input data for inference."""\n'

        if input_stream == "camera":
            code += "        if isinstance(input_data, str):\n"
            code += "            # Load image from file\n"
            code += "            image = cv2.imread(input_data)\n"
            code += "            if image is None:\n"
            code += (
                "                raise ValueError("
                'f"Could not load image: {input_data}")\n'
            )
            code += "        else:\n"
            code += "            image = input_data\n"
            code += "        \n"
            code += "        # Resize to model input size\n"
            code += "        input_shape = self.input_details[0]['shape']\n"
            code += "        height, width = input_shape[1], input_shape[2]\n"
            code += "        image = cv2.resize(image, (width, height))\n"
            code += "        \n"
            code += "        # Normalize to [0, 1]\n"
            code += "        image = image.astype(np.float32) / 255.0\n"
            code += "        \n"
            code += "        # Add batch dimension\n"
            code += "        image = np.expand_dims(image, axis=0)\n"
            code += "        \n"
            code += "        return image\n\n"
        else:
            code += "        # Generic preprocessing\n"
            code += "        if isinstance(input_data, str):\n"
            code += "            # Handle file input\n"
            code += "            data = np.load(input_data)\n"
            code += "        else:\n"
            code += "            data = input_data\n"
            code += "        \n"
            code += "        # Ensure correct shape\n"
            code += "        input_shape = self.input_details[0]['shape']\n"
            code += "        if data.shape != tuple(input_shape):\n"
            code += "            data = np.resize(data, input_shape)\n"
            code += "        \n"
            code += "        return data.astype(np.float32)\n\n"

        return code

    def _generate_python_inference_methods(self) -> str:
        """Generate inference methods."""
        code = (
            "    def predict(self, input_data: Union[np.ndarray, str]) -> np.ndarray:\n"
        )
        code += '        """Run inference on input data."""\n'
        code += "        start_time = time.time()\n"
        code += "        \n"
        code += "        # Preprocess input\n"
        code += "        processed_input = self._preprocess_input(input_data)\n"
        code += "        \n"
        code += "        # Set input tensor\n"
        code += "        self.interpreter.set_tensor(\n"
        code += "            self.input_details[0]['index'],\n"
        code += "            processed_input\n"
        code += "        )\n"
        code += "        \n"
        code += "        # Run inference\n"
        code += "        self.interpreter.invoke()\n"
        code += "        \n"
        code += "        # Get output\n"
        code += "        output = self.interpreter.get_tensor(\n"
        code += "            self.output_details[0]['index']\n"
        code += "        )\n"
        code += "        \n"
        code += "        inference_time = time.time() - start_time\n"
        code += '        print(f"Inference time: {inference_time:.4f}s")\n'
        code += "        \n"
        code += "        return output\n\n"

        # Batch inference
        code += (
            "    def predict_batch(self, input_data: "
            "List[Union[np.ndarray, str]]) -> List[np.ndarray]:\n"
        )
        code += '        """Run batch inference on multiple inputs."""\n'
        code += "        results = []\n"
        code += "        for data in input_data:\n"
        code += "            result = self.predict(data)\n"
        code += "            results.append(result)\n"
        code += "        return results\n\n"

        return code

    def _generate_python_utility_methods(self) -> str:
        """Generate utility methods."""
        code = "    def get_model_info(self) -> Dict[str, Any]:\n"
        code += '        """Get model information."""\n'
        code += "        return {\n"
        code += '            "model_path": self.model_path,\n'
        code += (
            "            \"input_shape\": self.input_details[0]['shape'] "
            "if self.input_details else None,\n"
        )
        code += (
            "            \"output_shape\": self.output_details[0]['shape'] "
            "if self.output_details else None,\n"
        )
        code += '            "buffer_size": self.buffer_size,\n'
        code += '            "memory_limit": self.memory_limit,\n'
        code += '            "optimize_for": self.optimize_for\n'
        code += "        }\n\n"

        code += (
            "    def benchmark(self, input_data: Union[np.ndarray, str], "
            "num_runs: int = 100) -> Dict[str, float]:\n"
        )
        code += '        """Benchmark inference performance."""\n'
        code += "        times = []\n"
        code += "        for _ in range(num_runs):\n"
        code += "            start = time.time()\n"
        code += "            self.predict(input_data)\n"
        code += "            times.append(time.time() - start)\n"
        code += "        \n"
        code += "        return {\n"
        code += '            "mean_time": np.mean(times),\n'
        code += '            "std_time": np.std(times),\n'
        code += '            "min_time": np.min(times),\n'
        code += '            "max_time": np.max(times)\n'
        code += "        }\n\n"

        return code

    def _generate_python_main(self) -> str:
        """Generate main execution code."""
        code = "def main():\n"
        code += '    """Main execution function."""\n'
        code += "    import argparse\n"
        code += "    \n"
        code += (
            '    parser = argparse.ArgumentParser(description="EdgeFlow Inference")\n'
        )
        code += (
            '    parser.add_argument("--input", required=True, '
            'help="Input data path")\n'
        )
        code += '    parser.add_argument("--model", help="Model path override")\n'
        code += (
            '    parser.add_argument("--benchmark", action="store_true", '
            'help="Run benchmark")\n'
        )
        code += "    args = parser.parse_args()\n"
        code += "    \n"
        code += "    # Initialize inference engine\n"
        code += "    engine = EdgeFlowInference(args.model)\n"
        code += "    \n"
        code += "    if args.benchmark:\n"
        code += "        # Run benchmark\n"
        code += "        results = engine.benchmark(args.input)\n"
        code += '        print("Benchmark Results:")\n'
        code += "        for key, value in results.items():\n"
        code += '            print(f"  {key}: {value:.4f}s")\n'
        code += "    else:\n"
        code += "        # Run inference\n"
        code += "        result = engine.predict(args.input)\n"
        code += '        print(f"Prediction result shape: {result.shape}")\n'
        code += '        print(f"Prediction result: {result}")\n\n'
        code += 'if __name__ == "__main__":\n'
        code += "    main()\n"

        return code

    def _generate_cpp_header(self) -> str:
        """Generate C++ header with includes and namespace."""
        code = "// EdgeFlow Generated C++ Inference Code\n"
        code += "// Generated from IR graph for bare-metal/embedded Linux\n\n"
        code += "#include <iostream>\n"
        code += "#include <vector>\n"
        code += "#include <memory>\n"
        code += "#include <chrono>\n"
        code += "#include <fstream>\n"
        code += "#include <cstring>\n\n"
        code += "// TensorFlow Lite C++ API\n"
        code += '#include "tensorflow/lite/interpreter.h"\n'
        code += '#include "tensorflow/lite/kernels/register.h"\n'
        code += '#include "tensorflow/lite/model.h"\n\n'
        code += "namespace edgeflow {\n\n"
        return code

    def _generate_cpp_includes(self) -> str:
        """Generate C++ includes for I/O handling."""
        code = "// I/O Handling includes\n"
        code += "#ifdef CAMERA_INPUT\n"
        code += "#include <opencv2/opencv.hpp>\n"
        code += "#endif\n\n"
        code += "#ifdef DISPLAY_OUTPUT\n"
        code += "#include <opencv2/opencv.hpp>\n"
        code += "#endif\n\n"
        code += "#ifdef NETWORK_STREAMING\n"
        code += "#include <zmq.hpp>\n"
        code += "#endif\n\n"
        return code

    def _generate_cpp_io_handling(self) -> str:
        """Generate C++ I/O handling code."""
        code = "class IOHandler {\n"
        code += "public:\n"
        code += "    IOHandler() = default;\n"
        code += "    virtual ~IOHandler() = default;\n\n"

        # Camera input
        code += "#ifdef CAMERA_INPUT\n"
        code += "    bool init_camera(int device_id = 0) {\n"
        code += "        cap.open(device_id);\n"
        code += "        return cap.isOpened();\n"
        code += "    }\n\n"
        code += "    std::vector<float> capture_frame() {\n"
        code += "        cv::Mat frame;\n"
        code += "        cap >> frame;\n"
        code += "        // Convert to model input format\n"
        code += "        return preprocess_frame(frame);\n"
        code += "    }\n\n"
        code += "private:\n"
        code += "    cv::VideoCapture cap;\n"
        code += "#endif\n\n"

        # Display output
        code += "#ifdef DISPLAY_OUTPUT\n"
        code += "    void display_result(const std::vector<float>& result) {\n"
        code += "        // Convert result to display format\n"
        code += "        cv::Mat display_frame = postprocess_result(result);\n"
        code += '        cv::imshow("EdgeFlow Output", display_frame);\n'
        code += "        cv::waitKey(1);\n"
        code += "    }\n"
        code += "#endif\n\n"

        # Network streaming
        code += "#ifdef NETWORK_STREAMING\n"
        code += "    bool init_network(const std::string& endpoint) {\n"
        code += "        context = std::make_unique<zmq::context_t>();\n"
        code += "        socket = std::make_unique<zmq::socket_t>(*context, ZMQ_PUB);\n"
        code += "        socket->bind(endpoint);\n"
        code += "        return true;\n"
        code += "    }\n\n"
        code += "    void stream_result(const std::vector<float>& result) {\n"
        code += "        // Serialize and send result\n"
        code += "        std::string data = serialize_result(result);\n"
        code += "        zmq::message_t message(data.size());\n"
        code += "        memcpy(message.data(), data.c_str(), data.size());\n"
        code += "        socket->send(message, zmq::send_flags::dontwait);\n"
        code += "    }\n\n"
        code += "private:\n"
        code += "    std::unique_ptr<zmq::context_t> context;\n"
        code += "    std::unique_ptr<zmq::socket_t> socket;\n"
        code += "#endif\n\n"

        code += "    std::vector<float> preprocess_frame(const cv::Mat& frame) {\n"
        code += "        // Implement frame preprocessing\n"
        code += "        return std::vector<float>();\n"
        code += "    }\n\n"
        code += "    cv::Mat postprocess_result(const std::vector<float>& result) {\n"
        code += "        // Implement result postprocessing\n"
        code += "        return cv::Mat();\n"
        code += "    }\n\n"
        code += "    std::string serialize_result(const std::vector<float>& result) {\n"
        code += "        // Implement result serialization\n"
        code += "        return std::string();\n"
        code += "    }\n"
        code += "};\n\n"
        return code

    def _generate_cpp_inference_engine(self) -> str:
        """Generate C++ inference engine from IR graph."""
        code = "class EdgeFlowInferenceEngine {\n"
        code += "public:\n"
        code += "    EdgeFlowInferenceEngine(const std::string& model_path) {\n"
        code += "        load_model(model_path);\n"
        code += "        io_handler = std::make_unique<IOHandler>();\n"
        code += "    }\n\n"

        # Model loading
        code += "    bool load_model(const std::string& model_path) {\n"
        code += "        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());\n"
        code += "        if (!model) return false;\n\n"
        code += "        tflite::ops::builtin::BuiltinOpResolver resolver;\n"
        code += "        tflite::InterpreterBuilder builder(*model, resolver);\n"
        code += "        builder(&interpreter);\n"
        code += "        if (!interpreter) return false;\n\n"
        code += "        interpreter->AllocateTensors();\n"
        code += "        return true;\n"
        code += "    }\n\n"

        # Inference method
        code += "    std::vector<float> predict(const std::vector<float>& input) {\n"
        code += "        auto start = std::chrono::high_resolution_clock::now();\n\n"
        code += "        // Copy input to model\n"
        code += (
            "        float* input_ptr = interpreter->typed_input_tensor<float>(0);\n"
        )
        code += "        std::memcpy(input_ptr, input.data(), input.size() * sizeof(float));\n\n"
        code += "        // Run inference\n"
        code += "        interpreter->Invoke();\n\n"
        code += "        // Get output\n"
        code += (
            "        float* output_ptr = interpreter->typed_output_tensor<float>(0);\n"
        )
        code += "        int output_size = interpreter->output_tensor(0)->bytes / sizeof(float);\n"
        code += "        std::vector<float> result(output_ptr, output_ptr + output_size);\n\n"
        code += "        auto end = std::chrono::high_resolution_clock::now();\n"
        code += "        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);\n"
        code += '        std::cout << "Inference time: " << duration.count() << " μs" << std::endl;\n\n'
        code += "        return result;\n"
        code += "    }\n\n"

        code += "private:\n"
        code += "    std::unique_ptr<tflite::FlatBufferModel> model;\n"
        code += "    std::unique_ptr<tflite::Interpreter> interpreter;\n"
        code += "    std::unique_ptr<IOHandler> io_handler;\n"
        code += "};\n\n"
        return code

    def _generate_cpp_main(self) -> str:
        """Generate C++ main function."""
        code = "} // namespace edgeflow\n\n"
        code += "int main(int argc, char* argv[]) {\n"
        code += "    if (argc < 2) {\n"
        code += '        std::cerr << "Usage: " << argv[0] << " <model_path> [input_data]" << std::endl;\n'
        code += "        return 1;\n"
        code += "    }\n\n"
        code += "    std::string model_path = argv[1];\n"
        code += "    edgeflow::EdgeFlowInferenceEngine engine(model_path);\n\n"
        code += "    // Initialize I/O\n"
        code += "#ifdef CAMERA_INPUT\n"
        code += "    if (!engine.io_handler->init_camera()) {\n"
        code += '        std::cerr << "Failed to initialize camera" << std::endl;\n'
        code += "        return 1;\n"
        code += "    }\n"
        code += "#endif\n\n"
        code += "    // Run inference loop\n"
        code += "    while (true) {\n"
        code += "#ifdef CAMERA_INPUT\n"
        code += "        auto input = engine.io_handler->capture_frame();\n"
        code += "#else\n"
        code += "        // Use dummy input for testing\n"
        code += "        std::vector<float> input(224 * 224 * 3, 0.5f);\n"
        code += "#endif\n\n"
        code += "        auto result = engine.predict(input);\n\n"
        code += "#ifdef DISPLAY_OUTPUT\n"
        code += "        engine.io_handler->display_result(result);\n"
        code += "#endif\n\n"
        code += "#ifdef NETWORK_STREAMING\n"
        code += "        engine.io_handler->stream_result(result);\n"
        code += "#endif\n\n"
        code += "        // Break on key press or after N iterations\n"
        code += "        static int iterations = 0;\n"
        code += "        if (++iterations > 100) break;\n"
        code += "    }\n\n"
        code += "    return 0;\n"
        code += "}\n"
        return code

    def _generate_onnx_inference_class(self) -> str:
        """Generate ONNX Runtime inference class."""
        code = "class EdgeFlowONNXInference:\n"
        code += '    """EdgeFlow inference engine using ONNX Runtime."""\n\n'
        code += "    def __init__(self, model_path: str):\n"
        code += "        self.model_path = model_path\n"
        code += "        self.session = None\n"
        code += "        self.input_name = None\n"
        code += "        self.output_name = None\n"
        code += "        self._load_model()\n\n"
        code += "    def _load_model(self):\n"
        code += '        """Load ONNX model."""\n'
        code += '        providers = ["CPUExecutionProvider"]\n'
        code += '        if ort.get_device() == "GPU":\n'
        code += '            providers.insert(0, "CUDAExecutionProvider")\n'
        code += "        \n"
        code += "        self.session = ort.InferenceSession(self.model_path, providers=providers)\n"
        code += "        self.input_name = self.session.get_inputs()[0].name\n"
        code += "        self.output_name = self.session.get_outputs()[0].name\n\n"
        code += "    def predict(self, input_data: np.ndarray) -> np.ndarray:\n"
        code += '        """Run inference."""\n'
        code += "        return self.session.run([self.output_name], {self.input_name: input_data})[0]\n\n"
        return code

    def _generate_tensorrt_inference_class(self) -> str:
        """Generate TensorRT inference class."""
        code = "class EdgeFlowTensorRTInference:\n"
        code += '    """EdgeFlow inference engine using TensorRT."""\n\n'
        code += "    def __init__(self, model_path: str):\n"
        code += "        self.model_path = model_path\n"
        code += "        self.engine = None\n"
        code += "        self.context = None\n"
        code += "        self._load_engine()\n\n"
        code += "    def _load_engine(self):\n"
        code += '        """Load TensorRT engine."""\n'
        code += "        logger = trt.Logger(trt.Logger.WARNING)\n"
        code += '        with open(self.model_path, "rb") as f:\n'
        code += "            runtime = trt.Runtime(logger)\n"
        code += "            self.engine = runtime.deserialize_cuda_engine(f.read())\n"
        code += "        self.context = self.engine.create_execution_context()\n\n"
        code += "    def predict(self, input_data: np.ndarray) -> np.ndarray:\n"
        code += '        """Run inference."""\n'
        code += "        # Allocate GPU memory\n"
        code += "        input_shape = self.engine.get_binding_shape(0)\n"
        code += "        output_shape = self.engine.get_binding_shape(1)\n"
        code += "        \n"
        code += "        # Run inference (simplified)\n"
        code += "        return np.zeros(output_shape, dtype=np.float32)\n\n"
        return code

    def _generate_tvm_inference_class(self) -> str:
        """Generate TVM inference class."""
        code = "class EdgeFlowTVMInference:\n"
        code += '    """EdgeFlow inference engine using TVM."""\n\n'
        code += "    def __init__(self, model_path: str):\n"
        code += "        self.model_path = model_path\n"
        code += "        self.module = None\n"
        code += "        self.device = None\n"
        code += "        self._load_model()\n\n"
        code += "    def _load_model(self):\n"
        code += '        """Load TVM model."""\n'
        code += "        # Load compiled module\n"
        code += "        self.module = tvm.runtime.load_module(self.model_path)\n"
        code += "        self.device = tvm.cpu(0)\n\n"
        code += "    def predict(self, input_data: np.ndarray) -> np.ndarray:\n"
        code += '        """Run inference."""\n'
        code += "        # Convert input to TVM tensor\n"
        code += "        tvm_input = tvm.nd.array(input_data, device=self.device)\n"
        code += "        \n"
        code += "        # Run inference\n"
        code += '        self.module["run"](tvm_input)\n'
        code += "        \n"
        code += "        # Get output (simplified)\n"
        code += "        return np.zeros((1, 1000), dtype=np.float32)\n\n"
        return code

    def _generate_cpp_code(self) -> str:
        """Generate C++ inference code."""
        # This is a simplified C++ generator
        # In a real implementation, this would be much more comprehensive

        code = "// EdgeFlow Generated C++ Inference Code\n"
        code += "// This file was automatically generated by EdgeFlow DSL compiler\n\n"

        code += "#include <tensorflow/lite/interpreter.h>\n"
        code += "#include <tensorflow/lite/kernels/register.h>\n"
        code += "#include <tensorflow/lite/model.h>\n"
        code += "#include <opencv2/opencv.hpp>\n"
        code += "#include <iostream>\n"
        code += "#include <chrono>\n"
        code += "#include <memory>\n\n"

        code += "class EdgeFlowInference {\n"
        code += "private:\n"
        code += "    std::unique_ptr<tflite::Interpreter> interpreter_;\n"
        code += "    std::unique_ptr<tflite::FlatBufferModel> model_;\n"
        code += f'    int buffer_size_ = {self.config.get("buffer_size", 32)};\n'
        mem_limit = self.config.get("memory_limit", 64)
        code += f"    int memory_limit_ = {mem_limit} * 1024 * 1024;\n"
        code += "    \n"
        code += "public:\n"
        code += "    EdgeFlowInference(const std::string& model_path);\n"
        code += "    ~EdgeFlowInference() = default;\n"
        code += "    \n"
        code += "    bool LoadModel(const std::string& model_path);\n"
        code += "    cv::Mat PreprocessInput(const cv::Mat& input);\n"
        code += "    std::vector<float> Predict(const cv::Mat& input);\n"
        code += "    void Benchmark(const cv::Mat& input, int num_runs = 100);\n"
        code += "};\n\n"

        code += "// Implementation would go here...\n"
        code += "// (Simplified for brevity)\n"

        return code

    def _generate_optimization_report(self) -> str:
        """Generate optimization report."""
        report = "# EdgeFlow Optimization Report\n\n"
        report += "## Applied Optimizations\n\n"

        if self.optimizations:
            for i, opt in enumerate(self.optimizations, 1):
                report += f"{i}. {opt}\n"
        else:
            report += "No optimizations applied.\n"

        report += "\n## Configuration Summary\n\n"
        for key, value in self.config.items():
            report += f"- **{key}**: `{value}`\n"

        report += "\n## Generated Code Features\n\n"
        report += "- Memory-optimized inference engine\n"
        report += "- Device-specific optimizations\n"
        report += "- Batch processing support\n"
        report += "- Built-in benchmarking\n"
        report += "- Error handling and logging\n"

        return report


def generate_code(ast: Program, output_dir: str = "generated") -> Dict[str, str]:
    """Generate Python and C++ code from AST.

    Args:
        ast: EdgeFlow AST
        output_dir: Directory to save generated files

    Returns:
        Dictionary with generated code files
    """
    generator = CodeGenerator(ast)

    # Generate code
    python_code = generator.generate_python_inference()
    cpp_code = generator.generate_cpp_inference()
    report = generator.generate_optimization_report()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save files
    files = {}
    files["python"] = os.path.join(output_dir, "inference.py")
    files["cpp"] = os.path.join(output_dir, "inference.cpp")
    files["report"] = os.path.join(output_dir, "optimization_report.md")

    with open(files["python"], "w") as f:
        f.write(python_code)

    with open(files["cpp"], "w") as f:
        f.write(cpp_code)

    with open(files["report"], "w") as f:
        f.write(report)

    return files
