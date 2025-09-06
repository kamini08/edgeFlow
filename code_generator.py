"""Code Generator for EdgeFlow DSL.

This module generates Python and C++ inference code from EdgeFlow AST.
It implements the visitor pattern to traverse the AST and generate
optimized inference code for edge devices.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set
from edgeflow_ast import (
    ASTNode, ASTVisitor, Program, ModelStatement, QuantizeStatement,
    TargetDeviceStatement, DeployPathStatement, InputStreamStatement,
    BufferSizeStatement, OptimizeForStatement, MemoryLimitStatement,
    FusionStatement, ConditionalStatement, PipelineStatement,
    Literal, Identifier, BinaryExpression, UnaryExpression, Condition
)


class CodeGenerator(ASTVisitor):
    """Generates Python and C++ inference code from EdgeFlow AST."""
    
    def __init__(self, ast: Program):
        self.ast = ast
        self.config: Dict[str, Any] = {}
        self.imports: Set[str] = set()
        self.optimizations: List[str] = []
        
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
        self.config['model_path'] = node.path
        return None
    
    def visit_quantize_statement(self, node: QuantizeStatement) -> Any:
        self.config['quantization'] = node.quant_type
        if node.quant_type != 'none':
            self.optimizations.append(f"Applied {node.quant_type.upper()} quantization")
        return None
    
    def visit_target_device_statement(self, node: TargetDeviceStatement) -> Any:
        self.config['target_device'] = node.device
        return None
    
    def visit_deploy_path_statement(self, node: DeployPathStatement) -> Any:
        self.config['deploy_path'] = node.path
        return None
    
    def visit_input_stream_statement(self, node: InputStreamStatement) -> Any:
        self.config['input_stream'] = node.stream
        return None
    
    def visit_buffer_size_statement(self, node: BufferSizeStatement) -> Any:
        self.config['buffer_size'] = node.size
        self.optimizations.append(f"Set buffer size to {node.size} for optimal streaming")
        return None
    
    def visit_optimize_for_statement(self, node: OptimizeForStatement) -> Any:
        self.config['optimize_for'] = node.goal
        self.optimizations.append(f"Optimized for {node.goal}")
        return None
    
    def visit_memory_limit_statement(self, node: MemoryLimitStatement) -> Any:
        self.config['memory_limit'] = node.limit_mb
        self.optimizations.append(f"Memory limit set to {node.limit_mb}MB")
        return None
    
    def visit_fusion_statement(self, node: FusionStatement) -> Any:
        self.config['enable_fusion'] = node.enabled
        if node.enabled:
            self.optimizations.append("Enabled operation fusion")
        return None
    
    def visit_conditional_statement(self, node: ConditionalStatement) -> Any:
        # Handle conditional logic
        return None
    
    def visit_pipeline_statement(self, node: PipelineStatement) -> Any:
        self.config['pipeline_steps'] = node.steps
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
        self.imports.update([
            "import tensorflow as tf",
            "import numpy as np",
            "import cv2",
            "import time",
            "from typing import Optional, Union, List, Tuple"
        ])
        
        # Add device-specific imports
        if self.config.get('target_device') == 'raspberry_pi':
            self.imports.add("import picamera")
        
        # Generate the Python class
        code = self._generate_python_header()
        code += self._generate_python_class()
        code += self._generate_python_main()
        
        return code
    
    def _generate_python_header(self) -> str:
        """Generate Python file header with imports and docstring."""
        header = '"""EdgeFlow Generated Python Inference Code.\n\n'
        header += 'This file was automatically generated by EdgeFlow DSL compiler.\n'
        header += 'Do not edit manually - changes will be overwritten.\n\n'
        header += 'Configuration:\n'
        for key, value in self.config.items():
            header += f'  {key}: {value}\n'
        header += '"""\n\n'
        
        # Add imports
        for imp in sorted(self.imports):
            header += imp + '\n'
        header += '\n'
        
        return header
    
    def _generate_python_class(self) -> str:
        """Generate the main Python inference class."""
        class_name = "EdgeFlowInference"
        model_path = self.config.get('model_path', 'model.tflite')
        quant_type = self.config.get('quantization', 'none')
        buffer_size = self.config.get('buffer_size', 32)
        memory_limit = self.config.get('memory_limit', 64)
        optimize_for = self.config.get('optimize_for', 'latency')
        
        code = f"class {class_name}:\n"
        code += '    """EdgeFlow inference engine for edge devices."""\n\n'
        
        # Constructor
        code += "    def __init__(self, model_path: str = None):\n"
        code += f'        """Initialize the inference engine."""\n'
        code += f'        self.model_path = model_path or "{model_path}"\n'
        code += f'        self.buffer_size = {buffer_size}\n'
        code += f'        self.memory_limit = {memory_limit} * 1024 * 1024  # Convert MB to bytes\n'
        code += f'        self.optimize_for = "{optimize_for}"\n'
        code += '        self.interpreter = None\n'
        code += '        self.input_details = None\n'
        code += '        self.output_details = None\n'
        code += '        self._setup_memory_management()\n'
        code += '        self._load_model()\n\n'
        
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
        code = '    def _setup_memory_management(self):\n'
        code += '        """Setup memory management for edge device."""\n'
        code += '        import gc\n'
        code += '        gc.set_threshold(100, 10, 10)  # Aggressive garbage collection\n'
        code += '        \n'
        code += '        # Set TensorFlow memory growth\n'
        code += '        gpus = tf.config.experimental.list_physical_devices(\'GPU\')\n'
        code += '        if gpus:\n'
        code += '            try:\n'
        code += '                for gpu in gpus:\n'
        code += '                    tf.config.experimental.set_memory_growth(gpu, True)\n'
        code += '            except RuntimeError as e:\n'
        code += '                print(f"GPU memory growth setup failed: {e}")\n\n'
        return code
    
    def _generate_python_model_loading(self, quant_type: str) -> str:
        """Generate model loading code."""
        code = '    def _load_model(self):\n'
        code += '        """Load and configure the TensorFlow Lite model."""\n'
        code += '        try:\n'
        code += '            # Load TFLite model\n'
        code += '            self.interpreter = tf.lite.Interpreter(\n'
        code += '                model_path=self.model_path,\n'
        code += '                experimental_preserve_all_tensors=True\n'
        code += '            )\n'
        code += '            self.interpreter.allocate_tensors()\n'
        code += '            \n'
        code += '            # Get input and output details\n'
        code += '            self.input_details = self.interpreter.get_input_details()\n'
        code += '            self.output_details = self.interpreter.get_output_details()\n'
        code += '            \n'
        code += '            print(f"Model loaded: {self.model_path}")\n'
        code += '            print(f"Input shape: {self.input_details[0][\'shape\']}")\n'
        code += '            print(f"Output shape: {self.output_details[0][\'shape\']}")\n'
        
        if quant_type != 'none':
            code += f'            print(f"Quantization: {quant_type.upper()}")\n'
        
        code += '        except Exception as e:\n'
        code += '            raise RuntimeError(f"Failed to load model: {e}")\n\n'
        return code
    
    def _generate_python_input_processing(self) -> str:
        """Generate input processing code."""
        input_stream = self.config.get('input_stream', 'camera')
        
        code = '    def _preprocess_input(self, input_data: Union[np.ndarray, str]) -> np.ndarray:\n'
        code += '        """Preprocess input data for inference."""\n'
        
        if input_stream == 'camera':
            code += '        if isinstance(input_data, str):\n'
            code += '            # Load image from file\n'
            code += '            image = cv2.imread(input_data)\n'
            code += '            if image is None:\n'
            code += '                raise ValueError(f"Could not load image: {input_data}")\n'
            code += '        else:\n'
            code += '            image = input_data\n'
            code += '        \n'
            code += '        # Resize to model input size\n'
            code += '        input_shape = self.input_details[0][\'shape\']\n'
            code += '        height, width = input_shape[1], input_shape[2]\n'
            code += '        image = cv2.resize(image, (width, height))\n'
            code += '        \n'
            code += '        # Normalize to [0, 1]\n'
            code += '        image = image.astype(np.float32) / 255.0\n'
            code += '        \n'
            code += '        # Add batch dimension\n'
            code += '        image = np.expand_dims(image, axis=0)\n'
            code += '        \n'
            code += '        return image\n\n'
        else:
            code += '        # Generic preprocessing\n'
            code += '        if isinstance(input_data, str):\n'
            code += '            # Handle file input\n'
            code += '            data = np.load(input_data)\n'
            code += '        else:\n'
            code += '            data = input_data\n'
            code += '        \n'
            code += '        # Ensure correct shape\n'
            code += '        input_shape = self.input_details[0][\'shape\']\n'
            code += '        if data.shape != tuple(input_shape):\n'
            code += '            data = np.resize(data, input_shape)\n'
            code += '        \n'
            code += '        return data.astype(np.float32)\n\n'
        
        return code
    
    def _generate_python_inference_methods(self) -> str:
        """Generate inference methods."""
        code = '    def predict(self, input_data: Union[np.ndarray, str]) -> np.ndarray:\n'
        code += '        """Run inference on input data."""\n'
        code += '        start_time = time.time()\n'
        code += '        \n'
        code += '        # Preprocess input\n'
        code += '        processed_input = self._preprocess_input(input_data)\n'
        code += '        \n'
        code += '        # Set input tensor\n'
        code += '        self.interpreter.set_tensor(\n'
        code += '            self.input_details[0][\'index\'],\n'
        code += '            processed_input\n'
        code += '        )\n'
        code += '        \n'
        code += '        # Run inference\n'
        code += '        self.interpreter.invoke()\n'
        code += '        \n'
        code += '        # Get output\n'
        code += '        output = self.interpreter.get_tensor(\n'
        code += '            self.output_details[0][\'index\']\n'
        code += '        )\n'
        code += '        \n'
        code += '        inference_time = time.time() - start_time\n'
        code += '        print(f"Inference time: {inference_time:.4f}s")\n'
        code += '        \n'
        code += '        return output\n\n'
        
        # Batch inference
        code += '    def predict_batch(self, input_data: List[Union[np.ndarray, str]]) -> List[np.ndarray]:\n'
        code += '        """Run batch inference on multiple inputs."""\n'
        code += '        results = []\n'
        code += '        for data in input_data:\n'
        code += '            result = self.predict(data)\n'
        code += '            results.append(result)\n'
        code += '        return results\n\n'
        
        return code
    
    def _generate_python_utility_methods(self) -> str:
        """Generate utility methods."""
        code = '    def get_model_info(self) -> Dict[str, Any]:\n'
        code += '        """Get model information."""\n'
        code += '        return {\n'
        code += '            "model_path": self.model_path,\n'
        code += '            "input_shape": self.input_details[0][\'shape\'] if self.input_details else None,\n'
        code += '            "output_shape": self.output_details[0][\'shape\'] if self.output_details else None,\n'
        code += '            "buffer_size": self.buffer_size,\n'
        code += '            "memory_limit": self.memory_limit,\n'
        code += '            "optimize_for": self.optimize_for\n'
        code += '        }\n\n'
        
        code += '    def benchmark(self, input_data: Union[np.ndarray, str], num_runs: int = 100) -> Dict[str, float]:\n'
        code += '        """Benchmark inference performance."""\n'
        code += '        times = []\n'
        code += '        for _ in range(num_runs):\n'
        code += '            start = time.time()\n'
        code += '            self.predict(input_data)\n'
        code += '            times.append(time.time() - start)\n'
        code += '        \n'
        code += '        return {\n'
        code += '            "mean_time": np.mean(times),\n'
        code += '            "std_time": np.std(times),\n'
        code += '            "min_time": np.min(times),\n'
        code += '            "max_time": np.max(times)\n'
        code += '        }\n\n'
        
        return code
    
    def _generate_python_main(self) -> str:
        """Generate main execution code."""
        code = 'def main():\n'
        code += '    """Main execution function."""\n'
        code += '    import argparse\n'
        code += '    \n'
        code += '    parser = argparse.ArgumentParser(description="EdgeFlow Inference")\n'
        code += '    parser.add_argument("--input", required=True, help="Input data path")\n'
        code += '    parser.add_argument("--model", help="Model path override")\n'
        code += '    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")\n'
        code += '    args = parser.parse_args()\n'
        code += '    \n'
        code += '    # Initialize inference engine\n'
        code += '    engine = EdgeFlowInference(args.model)\n'
        code += '    \n'
        code += '    if args.benchmark:\n'
        code += '        # Run benchmark\n'
        code += '        results = engine.benchmark(args.input)\n'
        code += '        print("Benchmark Results:")\n'
        code += '        for key, value in results.items():\n'
        code += '            print(f"  {key}: {value:.4f}s")\n'
        code += '    else:\n'
        code += '        # Run inference\n'
        code += '        result = engine.predict(args.input)\n'
        code += '        print(f"Prediction result shape: {result.shape}")\n'
        code += '        print(f"Prediction result: {result}")\n\n'
        code += 'if __name__ == "__main__":\n'
        code += '    main()\n'
        
        return code
    
    def _generate_cpp_code(self) -> str:
        """Generate C++ inference code."""
        # This is a simplified C++ generator
        # In a real implementation, this would be much more comprehensive
        
        code = '// EdgeFlow Generated C++ Inference Code\n'
        code += '// This file was automatically generated by EdgeFlow DSL compiler\n\n'
        
        code += '#include <tensorflow/lite/interpreter.h>\n'
        code += '#include <tensorflow/lite/kernels/register.h>\n'
        code += '#include <tensorflow/lite/model.h>\n'
        code += '#include <opencv2/opencv.hpp>\n'
        code += '#include <iostream>\n'
        code += '#include <chrono>\n'
        code += '#include <memory>\n\n'
        
        code += 'class EdgeFlowInference {\n'
        code += 'private:\n'
        code += '    std::unique_ptr<tflite::Interpreter> interpreter_;\n'
        code += '    std::unique_ptr<tflite::FlatBufferModel> model_;\n'
        code += f'    int buffer_size_ = {self.config.get("buffer_size", 32)};\n'
        code += f'    int memory_limit_ = {self.config.get("memory_limit", 64)} * 1024 * 1024;\n'
        code += '    \n'
        code += 'public:\n'
        code += '    EdgeFlowInference(const std::string& model_path);\n'
        code += '    ~EdgeFlowInference() = default;\n'
        code += '    \n'
        code += '    bool LoadModel(const std::string& model_path);\n'
        code += '    cv::Mat PreprocessInput(const cv::Mat& input);\n'
        code += '    std::vector<float> Predict(const cv::Mat& input);\n'
        code += '    void Benchmark(const cv::Mat& input, int num_runs = 100);\n'
        code += '};\n\n'
        
        code += '// Implementation would go here...\n'
        code += '// (Simplified for brevity)\n'
        
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
    files['python'] = os.path.join(output_dir, 'inference.py')
    files['cpp'] = os.path.join(output_dir, 'inference.cpp')
    files['report'] = os.path.join(output_dir, 'optimization_report.md')
    
    with open(files['python'], 'w') as f:
        f.write(python_code)
    
    with open(files['cpp'], 'w') as f:
        f.write(cpp_code)
    
    with open(files['report'], 'w') as f:
        f.write(report)
    
    return files

