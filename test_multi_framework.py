"""Comprehensive test suite for multi-framework support and interoperability.

This module provides comprehensive tests for the multi-framework support system,
including tests for framework parsers, UIR validation, optimization passes,
and end-to-end integration tests.
"""

import logging
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from unified_ir import (
    DataType,
    FrameworkType,
    OperationType,
    TensorInfo,
    TensorShape,
    UIRGraph,
    UIRNode,
    create_uir_from_edgeflow_config,
)
from framework_parsers import (
    TensorFlowParser,
    ONNXParser,
    PyTorchParser,
    FrameworkParserRegistry,
    parse_model_to_uir,
)
from uir_validators import (
    UIRValidationSuite,
    validate_uir_graph,
    ValidationResult,
    ValidationSeverity,
)
from uir_optimization_passes import (
    QuantizationPass,
    PruningPass,
    FusionPass,
    MemoryOptimizationPass,
    HardwareSpecificOptimizationPass,
    OptimizationPipeline,
    QuantizationType,
    create_optimization_pipeline,
)
from mlir_dialect import (
    UIRToMLIRConverter,
    MLIRPipeline,
    create_mlir_pipeline,
)

logger = logging.getLogger(__name__)


class TestUnifiedIR(unittest.TestCase):
    """Test cases for the unified intermediate representation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "model": "test_model.tflite",
            "quantize": "int8",
            "target_device": "raspberry_pi",
            "input_shape": "1,224,224,3",
        }
    
    def test_tensor_shape_creation(self):
        """Test tensor shape creation and operations."""
        # Test static shape
        shape = TensorShape([1, 224, 224, 3])
        self.assertEqual(shape.rank, 4)
        self.assertFalse(shape.is_dynamic())
        
        # Test dynamic shape
        dynamic_shape = TensorShape([1, -1, 224, 3])
        self.assertTrue(dynamic_shape.is_dynamic())
        
        # Test shape conversion
        static_converted = dynamic_shape.to_static({"batch_size": 2})
        self.assertEqual(static_converted.dimensions, [1, 2, 224, 3])
    
    def test_tensor_info_creation(self):
        """Test tensor info creation."""
        tensor = TensorInfo(
            name="input",
            shape=TensorShape([1, 224, 224, 3]),
            dtype=DataType.FLOAT32,
            framework_metadata={"test": True}
        )
        
        self.assertEqual(tensor.name, "input")
        self.assertEqual(tensor.dtype, DataType.FLOAT32)
        self.assertIn("test", tensor.framework_metadata)
    
    def test_uir_node_creation(self):
        """Test UIR node creation."""
        node = UIRNode(
            node_id="test_node",
            name="Test Node",
            operation_type=OperationType.CONV2D,
            framework_type=FrameworkType.TENSORFLOW,
            inputs=["input1", "input2"],
            outputs=["output1"],
        )
        
        node.add_attribute("kernel_size", 3)
        node.add_attribute("stride", 1)
        
        self.assertEqual(node.node_id, "test_node")
        self.assertEqual(node.operation_type, OperationType.CONV2D)
        self.assertEqual(node.get_attribute("kernel_size"), 3)
        self.assertEqual(node.get_attribute("stride"), 1)
    
    def test_uir_graph_creation(self):
        """Test UIR graph creation and operations."""
        graph = UIRGraph(
            name="test_graph",
            framework_type=FrameworkType.TENSORFLOW,
        )
        
        # Add tensors
        input_tensor = TensorInfo(
            name="input",
            shape=TensorShape([1, 224, 224, 3]),
            dtype=DataType.FLOAT32,
        )
        output_tensor = TensorInfo(
            name="output",
            shape=TensorShape([1, 1000]),
            dtype=DataType.FLOAT32,
        )
        
        graph.add_tensor(input_tensor)
        graph.add_tensor(output_tensor)
        
        # Add nodes
        conv_node = UIRNode(
            node_id="conv1",
            name="Conv1",
            operation_type=OperationType.CONV2D,
            framework_type=FrameworkType.TENSORFLOW,
            inputs=["input"],
            outputs=["output"],
        )
        
        graph.add_node(conv_node)
        graph.add_edge("input", "conv1", "input")
        graph.add_edge("conv1", "output", "output")
        
        # Validate graph
        is_valid, errors = graph.validate_graph()
        self.assertTrue(is_valid, f"Graph validation failed: {errors}")
        
        # Test graph info
        info = graph.get_graph_info()
        self.assertEqual(info["num_nodes"], 1)
        self.assertEqual(info["num_tensors"], 2)
        self.assertEqual(info["num_edges"], 2)
    
    def test_edgeflow_config_to_uir(self):
        """Test conversion from EdgeFlow config to UIR."""
        graph = create_uir_from_edgeflow_config(self.test_config)
        
        self.assertEqual(graph.name, "edgeflow_model")
        self.assertEqual(graph.framework_type, FrameworkType.TFLITE)
        self.assertIn("edgeflow_config", graph.framework_metadata)
        
        # Check that model node was created
        model_nodes = [node for node in graph.nodes.values() if node.operation_type == OperationType.CUSTOM]
        self.assertEqual(len(model_nodes), 1)
        
        model_node = model_nodes[0]
        self.assertIn("edgeflow_optimizations", model_node.framework_metadata)


class TestFrameworkParsers(unittest.TestCase):
    """Test cases for framework parsers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser_registry = FrameworkParserRegistry()
    
    def test_tensorflow_parser_initialization(self):
        """Test TensorFlow parser initialization."""
        parser = TensorFlowParser()
        self.assertIsInstance(parser, TensorFlowParser)
        self.assertEqual(parser.get_framework_type(), FrameworkType.TENSORFLOW)
        self.assertIn('.h5', parser.get_supported_formats())
        self.assertIn('.keras', parser.get_supported_formats())
    
    def test_onnx_parser_initialization(self):
        """Test ONNX parser initialization."""
        parser = ONNXParser()
        self.assertIsInstance(parser, ONNXParser)
        self.assertEqual(parser.get_framework_type(), FrameworkType.ONNX)
        self.assertIn('.onnx', parser.get_supported_formats())
    
    def test_pytorch_parser_initialization(self):
        """Test PyTorch parser initialization."""
        parser = PyTorchParser()
        self.assertIsInstance(parser, PyTorchParser)
        self.assertEqual(parser.get_framework_type(), FrameworkType.PYTORCH)
        self.assertIn('.pth', parser.get_supported_formats())
    
    def test_parser_registry(self):
        """Test framework parser registry."""
        # Test getting parsers
        tf_parser = self.parser_registry.get_parser(FrameworkType.TENSORFLOW)
        self.assertIsInstance(tf_parser, TensorFlowParser)
        
        onnx_parser = self.parser_registry.get_parser(FrameworkType.ONNX)
        self.assertIsInstance(onnx_parser, ONNXParser)
        
        pytorch_parser = self.parser_registry.get_parser(FrameworkType.PYTORCH)
        self.assertIsInstance(pytorch_parser, PyTorchParser)
        
        # Test supported formats
        formats = self.parser_registry.get_supported_formats()
        self.assertIn('.h5', formats)
        self.assertIn('.onnx', formats)
        self.assertIn('.pth', formats)
    
    def test_parse_model_simulation(self):
        """Test model parsing in simulation mode."""
        # Test TensorFlow model parsing
        tf_graph = parse_model_to_uir("test_model.h5")
        self.assertEqual(tf_graph.framework_type, FrameworkType.TENSORFLOW)
        self.assertIn("simulation_mode", tf_graph.framework_metadata)
        
        # Test ONNX model parsing
        onnx_graph = parse_model_to_uir("test_model.onnx")
        self.assertEqual(onnx_graph.framework_type, FrameworkType.ONNX)
        self.assertIn("simulation_mode", onnx_graph.framework_metadata)
        
        # Test PyTorch model parsing
        pytorch_graph = parse_model_to_uir("test_model.pth")
        self.assertEqual(pytorch_graph.framework_type, FrameworkType.PYTORCH)
        self.assertIn("simulation_mode", pytorch_graph.framework_metadata)
    
    def test_unsupported_format_error(self):
        """Test error handling for unsupported formats."""
        with self.assertRaises(ValueError):
            parse_model_to_uir("test_model.unsupported")


class TestUIRValidators(unittest.TestCase):
    """Test cases for UIR validators."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validation_suite = UIRValidationSuite()
        
        # Create a test graph
        self.test_graph = UIRGraph(
            name="test_graph",
            framework_type=FrameworkType.TENSORFLOW,
            framework_metadata={
                "target_device": "raspberry_pi",
                "quantize": "int8",
            }
        )
        
        # Add test tensors
        input_tensor = TensorInfo(
            name="input",
            shape=TensorShape([1, 224, 224, 3]),
            dtype=DataType.FLOAT32,
        )
        output_tensor = TensorInfo(
            name="output",
            shape=TensorShape([1, 1000]),
            dtype=DataType.FLOAT32,
        )
        
        self.test_graph.add_tensor(input_tensor)
        self.test_graph.add_tensor(output_tensor)
        
        # Add test nodes
        conv_node = UIRNode(
            node_id="conv1",
            name="Conv1",
            operation_type=OperationType.CONV2D,
            framework_type=FrameworkType.TENSORFLOW,
            inputs=["input"],
            outputs=["output"],
        )
        
        self.test_graph.add_node(conv_node)
        self.test_graph.add_edge("input", "conv1", "input")
        self.test_graph.add_edge("conv1", "output", "output")
    
    def test_validation_suite_initialization(self):
        """Test validation suite initialization."""
        self.assertIsInstance(self.validation_suite, UIRValidationSuite)
        self.assertGreater(len(self.validation_suite.validators), 0)
    
    def test_graph_validation(self):
        """Test graph validation."""
        result = self.validation_suite.validate_graph(self.test_graph)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertIsInstance(result.is_valid, bool)
        self.assertIsInstance(result.issues, list)
        self.assertIsInstance(result.compatibility_score, float)
        self.assertIsInstance(result.device_readiness, dict)
    
    def test_validate_uir_graph_function(self):
        """Test the validate_uir_graph function."""
        result = validate_uir_graph(self.test_graph)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertIsInstance(result.is_valid, bool)
    
    def test_validation_with_invalid_graph(self):
        """Test validation with an invalid graph."""
        # Create an invalid graph with missing tensor reference
        invalid_graph = UIRGraph(
            name="invalid_graph",
            framework_type=FrameworkType.TENSORFLOW,
        )
        
        # Add a node that references a non-existent tensor
        invalid_node = UIRNode(
            node_id="invalid_node",
            name="Invalid Node",
            operation_type=OperationType.CONV2D,
            framework_type=FrameworkType.TENSORFLOW,
            inputs=["non_existent_tensor"],
            outputs=["output"],
        )
        
        invalid_graph.add_node(invalid_node)
        invalid_graph.add_edge("non_existent_tensor", "invalid_node", "non_existent_tensor")
        
        result = validate_uir_graph(invalid_graph)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.issues), 0)


class TestOptimizationPasses(unittest.TestCase):
    """Test cases for optimization passes."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test graph
        self.test_graph = UIRGraph(
            name="test_graph",
            framework_type=FrameworkType.TENSORFLOW,
        )
        
        # Add test tensors
        input_tensor = TensorInfo(
            name="input",
            shape=TensorShape([1, 224, 224, 3]),
            dtype=DataType.FLOAT32,
        )
        output_tensor = TensorInfo(
            name="output",
            shape=TensorShape([1, 1000]),
            dtype=DataType.FLOAT32,
        )
        
        self.test_graph.add_tensor(input_tensor)
        self.test_graph.add_tensor(output_tensor)
        
        # Add test nodes
        conv_node = UIRNode(
            node_id="conv1",
            name="Conv1",
            operation_type=OperationType.CONV2D,
            framework_type=FrameworkType.TENSORFLOW,
            inputs=["input"],
            outputs=["output"],
        )
        
        self.test_graph.add_node(conv_node)
        self.test_graph.add_edge("input", "conv1", "input")
        self.test_graph.add_edge("conv1", "output", "output")
    
    def test_quantization_pass(self):
        """Test quantization pass."""
        # Test INT8 quantization
        int8_pass = QuantizationPass(QuantizationType.INT8)
        int8_graph = int8_pass.transform(self.test_graph)
        
        self.assertIn("quantized", int8_graph.name)
        self.assertIn("quantization_type", int8_graph.framework_metadata)
        self.assertEqual(int8_graph.framework_metadata["quantization_type"], "int8")
        
        # Test FLOAT16 quantization
        float16_pass = QuantizationPass(QuantizationType.FLOAT16)
        float16_graph = float16_pass.transform(self.test_graph)
        
        self.assertIn("quantized", float16_graph.name)
        self.assertEqual(float16_graph.framework_metadata["quantization_type"], "float16")
        
        # Test NO quantization
        none_pass = QuantizationPass(QuantizationType.NONE)
        none_graph = none_pass.transform(self.test_graph)
        
        self.assertEqual(none_graph.name, self.test_graph.name)  # Should be unchanged
    
    def test_pruning_pass(self):
        """Test pruning pass."""
        pruning_pass = PruningPass(sparsity=0.5, structured=True)
        pruned_graph = pruning_pass.transform(self.test_graph)
        
        self.assertIn("pruned", pruned_graph.name)
        self.assertIn("pruning_sparsity", pruned_graph.framework_metadata)
        self.assertEqual(pruned_graph.framework_metadata["pruning_sparsity"], 0.5)
        
        # Check that nodes have pruning attributes
        for node in pruned_graph.nodes.values():
            if node.operation_type in [OperationType.CONV2D, OperationType.DENSE]:
                self.assertIn("sparsity", node.attributes)
                self.assertIn("pruned", node.framework_metadata)
    
    def test_fusion_pass(self):
        """Test fusion pass."""
        # Create a graph with fusion opportunities
        fusion_graph = UIRGraph(
            name="fusion_test",
            framework_type=FrameworkType.TENSORFLOW,
        )
        
        # Add tensors
        input_tensor = TensorInfo(
            name="input",
            shape=TensorShape([1, 224, 224, 3]),
            dtype=DataType.FLOAT32,
        )
        hidden_tensor = TensorInfo(
            name="hidden",
            shape=TensorShape([1, 112, 112, 64]),
            dtype=DataType.FLOAT32,
        )
        output_tensor = TensorInfo(
            name="output",
            shape=TensorShape([1, 1000]),
            dtype=DataType.FLOAT32,
        )
        
        fusion_graph.add_tensor(input_tensor)
        fusion_graph.add_tensor(hidden_tensor)
        fusion_graph.add_tensor(output_tensor)
        
        # Add nodes that can be fused
        conv_node = UIRNode(
            node_id="conv1",
            name="Conv1",
            operation_type=OperationType.CONV2D,
            framework_type=FrameworkType.TENSORFLOW,
            inputs=["input"],
            outputs=["hidden"],
        )
        
        relu_node = UIRNode(
            node_id="relu1",
            name="ReLU1",
            operation_type=OperationType.RELU,
            framework_type=FrameworkType.TENSORFLOW,
            inputs=["hidden"],
            outputs=["output"],
        )
        
        fusion_graph.add_node(conv_node)
        fusion_graph.add_node(relu_node)
        fusion_graph.add_edge("input", "conv1", "input")
        fusion_graph.add_edge("conv1", "relu1", "hidden")
        fusion_graph.add_edge("relu1", "output", "output")
        
        # Apply fusion pass
        fusion_pass = FusionPass()
        fused_graph = fusion_pass.transform(fusion_graph)
        
        self.assertIn("fused", fused_graph.name)
        self.assertIn("fusion_applied", fused_graph.framework_metadata)
    
    def test_memory_optimization_pass(self):
        """Test memory optimization pass."""
        memory_pass = MemoryOptimizationPass()
        mem_opt_graph = memory_pass.transform(self.test_graph)
        
        self.assertIn("mem_opt", mem_opt_graph.name)
        self.assertIn("memory_optimized", mem_opt_graph.framework_metadata)
        
        # Check that nodes have memory optimization attributes
        for node in mem_opt_graph.nodes.values():
            self.assertIn("memory_optimized", node.framework_metadata)
            self.assertIn("memory_layout", node.attributes)
    
    def test_hardware_specific_pass(self):
        """Test hardware-specific optimization pass."""
        hw_pass = HardwareSpecificOptimizationPass("raspberry_pi")
        hw_graph = hw_pass.transform(self.test_graph)
        
        self.assertIn("hw_raspberry_pi", hw_graph.name)
        self.assertIn("hardware_optimized", hw_graph.framework_metadata)
        self.assertEqual(hw_graph.framework_metadata["target_device"], "raspberry_pi")
        
        # Check that nodes have hardware-specific attributes
        for node in hw_graph.nodes.values():
            self.assertIn("hardware_optimized", node.framework_metadata)
            self.assertIn("target_device", node.attributes)
    
    def test_optimization_pipeline(self):
        """Test optimization pipeline."""
        pipeline = OptimizationPipeline()
        
        # Add optimization passes
        pipeline.add_pass(QuantizationPass(QuantizationType.INT8))
        pipeline.add_pass(PruningPass(0.3))
        pipeline.add_pass(MemoryOptimizationPass())
        
        # Apply optimizations
        optimized_graph, results = pipeline.apply_optimizations(self.test_graph)
        
        self.assertNotEqual(optimized_graph.name, self.test_graph.name)
        self.assertEqual(len(results), 3)
        
        # Check that all passes succeeded
        for result in results:
            self.assertTrue(result.success)
    
    def test_create_optimization_pipeline(self):
        """Test creating a standard optimization pipeline."""
        pipeline = create_optimization_pipeline("raspberry_pi", QuantizationType.INT8, 0.3)
        
        self.assertIsInstance(pipeline, OptimizationPipeline)
        self.assertGreater(len(pipeline.passes), 0)


class TestMLIRDialect(unittest.TestCase):
    """Test cases for MLIR dialect."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.converter = UIRToMLIRConverter()
        
        # Create a test graph
        self.test_graph = UIRGraph(
            name="test_graph",
            framework_type=FrameworkType.TENSORFLOW,
        )
        
        # Add test tensors
        input_tensor = TensorInfo(
            name="input",
            shape=TensorShape([1, 224, 224, 3]),
            dtype=DataType.FLOAT32,
        )
        output_tensor = TensorInfo(
            name="output",
            shape=TensorShape([1, 1000]),
            dtype=DataType.FLOAT32,
        )
        
        self.test_graph.add_tensor(input_tensor)
        self.test_graph.add_tensor(output_tensor)
        
        # Add test nodes
        conv_node = UIRNode(
            node_id="conv1",
            name="Conv1",
            operation_type=OperationType.CONV2D,
            framework_type=FrameworkType.TENSORFLOW,
            inputs=["input"],
            outputs=["output"],
        )
        
        self.test_graph.add_node(conv_node)
        self.test_graph.add_edge("input", "conv1", "input")
        self.test_graph.add_edge("conv1", "output", "output")
    
    def test_uir_to_mlir_converter(self):
        """Test UIR to MLIR conversion."""
        mlir_module = self.converter.convert_to_mlir(self.test_graph)
        
        self.assertEqual(mlir_module.name, self.test_graph.name)
        self.assertEqual(len(mlir_module.operations), 1)
        self.assertIsNotNone(mlir_module.to_mlir_text())
    
    def test_mlir_pipeline(self):
        """Test MLIR compilation pipeline."""
        pipeline = create_mlir_pipeline("raspberry_pi")
        
        self.assertIsInstance(pipeline, MLIRPipeline)
        self.assertGreater(len(pipeline.passes), 0)
        
        # Compile through pipeline
        mlir_module, optimized_graph = pipeline.compile(self.test_graph, "raspberry_pi")
        
        self.assertIsNotNone(mlir_module)
        self.assertIsNotNone(optimized_graph)
        self.assertNotEqual(optimized_graph.name, self.test_graph.name)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete multi-framework pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_configs = [
            {
                "model": "test_model.tflite",
                "quantize": "int8",
                "target_device": "raspberry_pi",
                "input_shape": "1,224,224,3",
            },
            {
                "model": "test_model.onnx",
                "quantize": "float16",
                "target_device": "jetson_nano",
                "input_shape": "1,3,224,224",
            },
            {
                "model": "test_model.h5",
                "quantize": "none",
                "target_device": "cpu",
                "input_shape": "1,224,224,3",
            },
        ]
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        for config in self.test_configs:
            with self.subTest(config=config):
                # Step 1: Create UIR from EdgeFlow config
                graph = create_uir_from_edgeflow_config(config)
                self.assertIsInstance(graph, UIRGraph)
                
                # Step 2: Validate UIR graph
                validation_result = validate_uir_graph(graph)
                self.assertIsInstance(validation_result, ValidationResult)
                
                # Step 3: Apply optimizations
                optimization_pipeline = create_optimization_pipeline(
                    config["target_device"],
                    QuantizationType.INT8 if config["quantize"] == "int8" else QuantizationType.FLOAT16,
                    0.3
                )
                optimized_graph, optimization_results = optimization_pipeline.apply_optimizations(graph)
                self.assertIsInstance(optimized_graph, UIRGraph)
                self.assertIsInstance(optimization_results, list)
                
                # Step 4: Convert to MLIR
                mlir_pipeline = create_mlir_pipeline(config["target_device"])
                mlir_module, final_graph = mlir_pipeline.compile(optimized_graph, config["target_device"])
                self.assertIsNotNone(mlir_module)
                self.assertIsNotNone(final_graph)
                
                # Verify that the pipeline preserved important information
                self.assertIn("target_device", final_graph.framework_metadata)
                self.assertEqual(final_graph.framework_metadata["target_device"], config["target_device"])
    
    def test_framework_parser_integration(self):
        """Test integration with framework parsers."""
        test_models = [
            "test_model.tflite",
            "test_model.onnx",
            "test_model.h5",
            "test_model.pth",
        ]
        
        for model_path in test_models:
            with self.subTest(model=model_path):
                # Parse model to UIR
                graph = parse_model_to_uir(model_path)
                self.assertIsInstance(graph, UIRGraph)
                
                # Validate the parsed graph
                validation_result = validate_uir_graph(graph)
                self.assertIsInstance(validation_result, ValidationResult)
                
                # Apply optimizations
                optimization_pipeline = create_optimization_pipeline("cpu")
                optimized_graph, results = optimization_pipeline.apply_optimizations(graph)
                self.assertIsInstance(optimized_graph, UIRGraph)
    
    def test_error_handling(self):
        """Test error handling in the pipeline."""
        # Test with invalid model path
        with self.assertRaises(ValueError):
            parse_model_to_uir("invalid_model.unsupported")
        
        # Test with invalid configuration
        invalid_config = {
            "model": "test_model.tflite",
            "quantize": "invalid_quantization",
            "target_device": "invalid_device",
        }
        
        # This should not raise an exception, but should handle gracefully
        graph = create_uir_from_edgeflow_config(invalid_config)
        self.assertIsInstance(graph, UIRGraph)
        
        # Validation should catch issues
        validation_result = validate_uir_graph(graph)
        self.assertIsInstance(validation_result, ValidationResult)


def run_tests():
    """Run all tests."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestUnifiedIR,
        TestFrameworkParsers,
        TestUIRValidators,
        TestOptimizationPasses,
        TestMLIRDialect,
        TestIntegration,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

