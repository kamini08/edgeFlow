"""Framework-specific parsers for multi-framework support.

This module implements parsers that convert models from different ML frameworks
(TensorFlow, ONNX, PyTorch) into the unified intermediate representation (UIR).
Each parser handles framework-specific model formats and converts them to a
common representation that can be processed by EdgeFlow's optimization pipeline.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from unified_ir import (
    DataType,
    FrameworkParser,
    FrameworkType,
    OperationType,
    TensorInfo,
    TensorShape,
    UIRGraph,
    UIRNode,
)

logger = logging.getLogger(__name__)


class TensorFlowParser(FrameworkParser):
    """Parser for TensorFlow models (SavedModel, Keras, HDF5)."""

    def __init__(self):
        self.tf_available = False
        try:
            import tensorflow as tf

            self.tf = tf
            self.tf_available = True
            logger.info("TensorFlow parser initialized successfully")
        except ImportError:
            logger.warning("TensorFlow not available, parser will use simulation mode")

    def parse_model(self, model_path: str) -> UIRGraph:
        """Parse a TensorFlow model into UIR."""
        if not self.tf_available:
            return self._simulate_parsing(model_path)

        try:
            # Determine model type and load accordingly
            if os.path.isdir(model_path):
                # SavedModel format
                return self._parse_saved_model(model_path)
            elif model_path.endswith((".h5", ".keras")):
                # Keras model
                return self._parse_keras_model(model_path)
            else:
                raise ValueError(f"Unsupported TensorFlow model format: {model_path}")

        except Exception as e:
            logger.error(f"Failed to parse TensorFlow model {model_path}: {e}")
            return self._simulate_parsing(model_path)

    def _parse_saved_model(self, model_path: str) -> UIRGraph:
        """Parse a TensorFlow SavedModel."""
        model = self.tf.saved_model.load(model_path)

        graph = UIRGraph(
            name=os.path.basename(model_path),
            framework_type=FrameworkType.TENSORFLOW,
            framework_metadata={
                "model_path": model_path,
                "model_type": "saved_model",
                "signatures": list(model.signatures.keys())
                if hasattr(model, "signatures")
                else [],
            },
        )

        # Extract computational graph information
        concrete_func = model.signatures[list(model.signatures.keys())[0]]
        func_graph = concrete_func.graph

        # Parse nodes from the graph
        for op in func_graph.get_operations():
            node = self._convert_tf_operation_to_uir(op)
            graph.add_node(node)

        # Parse tensors
        for tensor in func_graph.get_tensor_by_name:
            tensor_info = self._convert_tf_tensor_to_uir(tensor)
            graph.add_tensor(tensor_info)

        return graph

    def _parse_keras_model(self, model_path: str) -> UIRGraph:
        """Parse a Keras model."""
        model = self.tf.keras.models.load_model(model_path)

        graph = UIRGraph(
            name=os.path.basename(model_path),
            framework_type=FrameworkType.TENSORFLOW,
            framework_metadata={
                "model_path": model_path,
                "model_type": "keras",
                "model_config": model.get_config()
                if hasattr(model, "get_config")
                else {},
            },
        )

        # Parse model layers
        for i, layer in enumerate(model.layers):
            node = self._convert_keras_layer_to_uir(layer, i)
            graph.add_node(node)

        # Parse input and output tensors
        if hasattr(model, "input_shape"):
            input_tensor = TensorInfo(
                name="input",
                shape=TensorShape(
                    model.input_shape[1:]
                    if model.input_shape[0] is None
                    else model.input_shape
                ),
                dtype=DataType.FLOAT32,
                framework_metadata={"keras_input": True},
            )
            graph.add_tensor(input_tensor)

        if hasattr(model, "output_shape"):
            output_tensor = TensorInfo(
                name="output",
                shape=TensorShape(
                    model.output_shape[1:]
                    if model.output_shape[0] is None
                    else model.output_shape
                ),
                dtype=DataType.FLOAT32,
                framework_metadata={"keras_output": True},
            )
            graph.add_tensor(output_tensor)

        return graph

    def _convert_tf_operation_to_uir(self, op) -> UIRNode:
        """Convert a TensorFlow operation to UIR node."""
        op_type = self._map_tf_op_to_uir_op(op.type)

        node = UIRNode(
            node_id=op.name,
            name=op.name,
            operation_type=op_type,
            framework_type=FrameworkType.TENSORFLOW,
            framework_metadata={
                "tf_op_type": op.type,
                "tf_device": op.device,
                "tf_control_inputs": [inp.name for inp in op.control_inputs],
            },
        )

        # Add operation attributes
        for attr_name, attr_value in op.node_def.attr.items():
            node.add_attribute(
                attr_name, attr_value, self._map_tf_attr_to_uir_dtype(attr_value)
            )

        return node

    def _convert_keras_layer_to_uir(self, layer, layer_index: int) -> UIRNode:
        """Convert a Keras layer to UIR node."""
        op_type = self._map_keras_layer_to_uir_op(type(layer).__name__)

        node = UIRNode(
            node_id=f"layer_{layer_index}",
            name=layer.name,
            operation_type=op_type,
            framework_type=FrameworkType.TENSORFLOW,
            framework_metadata={
                "keras_layer_type": type(layer).__name__,
                "keras_layer_config": layer.get_config()
                if hasattr(layer, "get_config")
                else {},
                "layer_index": layer_index,
            },
        )

        # Add layer configuration as attributes
        if hasattr(layer, "get_config"):
            config = layer.get_config()
            for key, value in config.items():
                if key not in ["name", "trainable"]:  # Skip common metadata
                    node.add_attribute(key, value)

        return node

    def _convert_tf_tensor_to_uir(self, tensor) -> TensorInfo:
        """Convert a TensorFlow tensor to UIR tensor info."""
        shape = TensorShape(
            [dim.value if dim.value is not None else -1 for dim in tensor.shape]
        )
        dtype = self._map_tf_dtype_to_uir_dtype(tensor.dtype)

        return TensorInfo(
            name=tensor.name,
            shape=shape,
            dtype=dtype,
            framework_metadata={
                "tf_tensor_name": tensor.name,
                "tf_dtype": str(tensor.dtype),
            },
        )

    def _map_tf_op_to_uir_op(self, tf_op_type: str) -> OperationType:
        """Map TensorFlow operation types to UIR operation types."""
        mapping = {
            "Conv2D": OperationType.CONV2D,
            "Conv1D": OperationType.CONV1D,
            "Conv3D": OperationType.CONV3D,
            "DepthwiseConv2dNative": OperationType.DEPTHWISE_CONV2D,
            "MaxPool": OperationType.MAX_POOL,
            "AvgPool": OperationType.AVG_POOL,
            "GlobalMaxPool": OperationType.GLOBAL_MAX_POOL,
            "GlobalAvgPool": OperationType.GLOBAL_AVG_POOL,
            "Relu": OperationType.RELU,
            "Sigmoid": OperationType.SIGMOID,
            "Tanh": OperationType.TANH,
            "Softmax": OperationType.SOFTMAX,
            "FusedBatchNorm": OperationType.BATCH_NORM,
            "MatMul": OperationType.MATMUL,
            "Reshape": OperationType.RESHAPE,
            "Transpose": OperationType.TRANSPOSE,
            "Add": OperationType.ADD,
            "Sub": OperationType.SUB,
            "Mul": OperationType.MUL,
            "Div": OperationType.DIV,
            "Sum": OperationType.REDUCE_SUM,
            "Mean": OperationType.REDUCE_MEAN,
            "Max": OperationType.REDUCE_MAX,
            "Min": OperationType.REDUCE_MIN,
            "ConcatV2": OperationType.CONCAT,
            "Split": OperationType.SPLIT,
        }
        return mapping.get(tf_op_type, OperationType.CUSTOM)

    def _map_keras_layer_to_uir_op(self, keras_layer_type: str) -> OperationType:
        """Map Keras layer types to UIR operation types."""
        mapping = {
            "Conv2D": OperationType.CONV2D,
            "Conv1D": OperationType.CONV1D,
            "Conv3D": OperationType.CONV3D,
            "DepthwiseConv2D": OperationType.DEPTHWISE_CONV2D,
            "SeparableConv2D": OperationType.SEPARABLE_CONV2D,
            "MaxPooling2D": OperationType.MAX_POOL,
            "AveragePooling2D": OperationType.AVG_POOL,
            "GlobalMaxPooling2D": OperationType.GLOBAL_MAX_POOL,
            "GlobalAveragePooling2D": OperationType.GLOBAL_AVG_POOL,
            "ReLU": OperationType.RELU,
            "Sigmoid": OperationType.SIGMOID,
            "Tanh": OperationType.TANH,
            "Softmax": OperationType.SOFTMAX,
            "GELU": OperationType.GELU,
            "Swish": OperationType.SWISH,
            "LeakyReLU": OperationType.LEAKY_RELU,
            "BatchNormalization": OperationType.BATCH_NORM,
            "LayerNormalization": OperationType.LAYER_NORM,
            "GroupNormalization": OperationType.GROUP_NORM,
            "Dense": OperationType.DENSE,
            "Reshape": OperationType.RESHAPE,
            "Flatten": OperationType.FLATTEN,
            "LSTM": OperationType.LSTM,
            "GRU": OperationType.GRU,
            "SimpleRNN": OperationType.RNN,
        }
        return mapping.get(keras_layer_type, OperationType.CUSTOM)

    def _map_tf_dtype_to_uir_dtype(self, tf_dtype) -> DataType:
        """Map TensorFlow data types to UIR data types."""
        mapping = {
            "float32": DataType.FLOAT32,
            "float16": DataType.FLOAT16,
            "int8": DataType.INT8,
            "int16": DataType.INT16,
            "int32": DataType.INT32,
            "int64": DataType.INT64,
            "uint8": DataType.UINT8,
            "uint16": DataType.UINT16,
            "uint32": DataType.UINT32,
            "uint64": DataType.UINT64,
            "bool": DataType.BOOL,
            "string": DataType.STRING,
            "complex64": DataType.COMPLEX64,
            "complex128": DataType.COMPLEX128,
        }
        return mapping.get(str(tf_dtype), DataType.FLOAT32)

    def _map_tf_attr_to_uir_dtype(self, attr_value) -> Optional[DataType]:
        """Map TensorFlow attribute values to UIR data types."""
        if hasattr(attr_value, "type"):
            return self._map_tf_dtype_to_uir_dtype(attr_value.type)
        return None

    def _simulate_parsing(self, model_path: str) -> UIRGraph:
        """Simulate parsing when TensorFlow is not available."""
        logger.warning(f"Simulating TensorFlow model parsing for {model_path}")

        graph = UIRGraph(
            name=os.path.basename(model_path),
            framework_type=FrameworkType.TENSORFLOW,
            framework_metadata={
                "model_path": model_path,
                "simulation_mode": True,
                "error": "TensorFlow not available",
            },
        )

        # Create a simple placeholder model
        input_tensor = TensorInfo(
            name="input",
            shape=TensorShape([1, 224, 224, 3]),
            dtype=DataType.FLOAT32,
            framework_metadata={"simulated": True},
        )

        output_tensor = TensorInfo(
            name="output",
            shape=TensorShape([1, 1000]),
            dtype=DataType.FLOAT32,
            framework_metadata={"simulated": True},
        )

        model_node = UIRNode(
            node_id="model",
            name="Simulated Model",
            operation_type=OperationType.CUSTOM,
            framework_type=FrameworkType.TENSORFLOW,
            framework_metadata={"simulated": True},
        )

        graph.add_tensor(input_tensor)
        graph.add_tensor(output_tensor)
        graph.add_node(model_node)
        graph.add_edge("input", "model", "input")
        graph.add_edge("model", "output", "output")

        return graph

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return [".h5", ".keras", ".pb", "saved_model"]

    def get_framework_type(self) -> FrameworkType:
        """Get the framework type."""
        return FrameworkType.TENSORFLOW


class ONNXParser(FrameworkParser):
    """Parser for ONNX models."""

    def __init__(self):
        self.onnx_available = False
        self._node_counter = 0
        try:
            import onnx

            self.onnx = onnx
            self.onnx_available = True
            logger.info("ONNX parser initialized successfully")
        except ImportError:
            logger.warning("ONNX not available, parser will use simulation mode")

    def parse_model(self, model_path: str) -> UIRGraph:
        """Parse an ONNX model into UIR."""
        if not self.onnx_available:
            return self._simulate_parsing(model_path)

        try:
            model = self.onnx.load(model_path)
            # Enrich model with inferred shapes when possible
            try:
                model = self.onnx.shape_inference.infer_shapes(model)
            except Exception as shape_err:
                logger.warning(f"ONNX shape inference failed: {shape_err}")

            graph = UIRGraph(
                name=os.path.basename(model_path),
                framework_type=FrameworkType.ONNX,
                framework_metadata={
                    "model_path": model_path,
                    "model_type": "onnx",
                    "ir_version": model.ir_version,
                    "opset_version": model.opset_import[0].version
                    if model.opset_import
                    else None,
                    "producer_name": model.producer_name,
                    "producer_version": model.producer_version,
                },
            )

            # Parse nodes from the graph
            for idx, node in enumerate(model.graph.node):
                uir_node = self._convert_onnx_node_to_uir(node, idx)
                graph.add_node(uir_node)

            # Parse input tensors
            for input_tensor in model.graph.input:
                tensor_info = self._convert_onnx_tensor_to_uir(input_tensor)
                graph.add_tensor(tensor_info)

            # Parse output tensors
            for output_tensor in model.graph.output:
                tensor_info = self._convert_onnx_tensor_to_uir(output_tensor)
                graph.add_tensor(tensor_info)

            # Parse initializer tensors (weights)
            for initializer in model.graph.initializer:
                tensor_info = self._convert_onnx_initializer_to_uir(initializer)
                graph.add_tensor(tensor_info)

            return graph

        except Exception as e:
            logger.error(f"Failed to parse ONNX model {model_path}: {e}")
            return self._simulate_parsing(model_path)

    def _convert_onnx_node_to_uir(self, node, index: int) -> UIRNode:
        """Convert an ONNX node to UIR node."""
        op_type = self._map_onnx_op_to_uir_op(node.op_type)
        # Robust ID: prefer node.name, else generate stable fallback
        node_name = (
            node.name if getattr(node, "name", None) else f"{node.op_type}_{index}"
        )

        uir_node = UIRNode(
            node_id=node_name,
            name=node_name,
            operation_type=op_type,
            framework_type=FrameworkType.ONNX,
            inputs=node.input,
            outputs=node.output,
            framework_metadata={
                "onnx_op_type": node.op_type,
                "onnx_domain": node.domain,
            },
        )

        # Add node attributes
        for attr in node.attribute:
            attr_value = self._extract_onnx_attr_value(attr)
            uir_node.add_attribute(
                attr.name, attr_value, self._map_onnx_attr_type_to_uir_dtype(attr.type)
            )

        return uir_node

    def _convert_onnx_tensor_to_uir(self, tensor) -> TensorInfo:
        """Convert an ONNX tensor to UIR tensor info."""
        shape = self._extract_onnx_tensor_shape(tensor)
        dtype = self._map_onnx_dtype_to_uir_dtype(tensor.type.tensor_type.elem_type)

        return TensorInfo(
            name=tensor.name,
            shape=shape,
            dtype=dtype,
            framework_metadata={
                "onnx_tensor_name": tensor.name,
                "onnx_dtype": tensor.type.tensor_type.elem_type,
            },
        )

    def _convert_onnx_initializer_to_uir(self, initializer) -> TensorInfo:
        """Convert an ONNX initializer to UIR tensor info."""
        shape = TensorShape(list(initializer.dims))
        dtype = self._map_onnx_dtype_to_uir_dtype(initializer.data_type)

        return TensorInfo(
            name=initializer.name,
            shape=shape,
            dtype=dtype,
            framework_metadata={
                "onnx_initializer": True,
                "onnx_dtype": initializer.data_type,
            },
        )

    def _extract_onnx_tensor_shape(self, tensor) -> TensorShape:
        """Extract shape from ONNX tensor."""
        shape_proto = tensor.type.tensor_type.shape
        dimensions = []

        for dim in shape_proto.dim:
            if dim.HasField("dim_value"):
                dimensions.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                dimensions.append(dim.dim_param)  # Dynamic dimension
            else:
                dimensions.append(-1)  # Unknown dimension

        return TensorShape(dimensions)

    def _extract_onnx_attr_value(self, attr) -> Any:
        """Extract value from ONNX attribute."""
        if attr.type == 1:  # FLOAT
            return attr.f
        elif attr.type == 2:  # INT
            return attr.i
        elif attr.type == 3:  # STRING
            return attr.s.decode("utf-8")
        elif attr.type == 4:  # TENSOR
            return attr.t
        elif attr.type == 5:  # GRAPH
            return attr.g
        elif attr.type == 6:  # FLOATS
            return list(attr.floats)
        elif attr.type == 7:  # INTS
            return list(attr.ints)
        elif attr.type == 8:  # STRINGS
            return [s.decode("utf-8") for s in attr.strings]
        elif attr.type == 9:  # TENSORS
            return list(attr.tensors)
        elif attr.type == 10:  # GRAPHS
            return list(attr.graphs)
        else:
            return None

    def _map_onnx_op_to_uir_op(self, onnx_op_type: str) -> OperationType:
        """Map ONNX operation types to UIR operation types."""
        mapping = {
            "Conv": OperationType.CONV2D,
            "ConvTranspose": OperationType.CONV2D,
            "MaxPool": OperationType.MAX_POOL,
            "AveragePool": OperationType.AVG_POOL,
            "GlobalMaxPool": OperationType.GLOBAL_MAX_POOL,
            "GlobalAveragePool": OperationType.GLOBAL_AVG_POOL,
            "Relu": OperationType.RELU,
            "Sigmoid": OperationType.SIGMOID,
            "Tanh": OperationType.TANH,
            "Softmax": OperationType.SOFTMAX,
            "Gelu": OperationType.GELU,
            "BatchNormalization": OperationType.BATCH_NORM,
            "LayerNormalization": OperationType.LAYER_NORM,
            "GroupNormalization": OperationType.GROUP_NORM,
            "Gemm": OperationType.MATMUL,
            "MatMul": OperationType.MATMUL,
            "Reshape": OperationType.RESHAPE,
            "Transpose": OperationType.TRANSPOSE,
            "Flatten": OperationType.FLATTEN,
            "Squeeze": OperationType.SQUEEZE,
            "Unsqueeze": OperationType.UNSQUEEZE,
            "Add": OperationType.ADD,
            "Sub": OperationType.SUB,
            "Mul": OperationType.MUL,
            "Div": OperationType.DIV,
            "Pow": OperationType.POW,
            "Sqrt": OperationType.SQRT,
            "Abs": OperationType.ABS,
            "ReduceSum": OperationType.REDUCE_SUM,
            "ReduceMean": OperationType.REDUCE_MEAN,
            "ReduceMax": OperationType.REDUCE_MAX,
            "ReduceMin": OperationType.REDUCE_MIN,
            "Concat": OperationType.CONCAT,
            "Split": OperationType.SPLIT,
            "LSTM": OperationType.LSTM,
            "GRU": OperationType.GRU,
            "Attention": OperationType.ATTENTION,
        }
        return mapping.get(onnx_op_type, OperationType.CUSTOM)

    def _map_onnx_dtype_to_uir_dtype(self, onnx_dtype: int) -> DataType:
        """Map ONNX data types to UIR data types."""
        mapping = {
            1: DataType.FLOAT32,  # FLOAT
            2: DataType.UINT8,  # UINT8
            3: DataType.INT8,  # INT8
            6: DataType.INT32,  # INT32
            7: DataType.INT64,  # INT64
            9: DataType.BOOL,  # BOOL
            10: DataType.FLOAT16,  # FLOAT16
            11: DataType.FLOAT64,  # DOUBLE
            12: DataType.UINT32,  # UINT32
            13: DataType.UINT64,  # UINT64
        }
        return mapping.get(onnx_dtype, DataType.FLOAT32)

    def _map_onnx_attr_type_to_uir_dtype(self, attr_type: int) -> Optional[DataType]:
        """Map ONNX attribute types to UIR data types."""
        if attr_type == 1:  # FLOAT
            return DataType.FLOAT32
        elif attr_type == 2:  # INT
            return DataType.INT64
        elif attr_type == 3:  # STRING
            return DataType.STRING
        return None

    def _simulate_parsing(self, model_path: str) -> UIRGraph:
        """Simulate parsing when ONNX is not available."""
        logger.warning(f"Simulating ONNX model parsing for {model_path}")

        graph = UIRGraph(
            name=os.path.basename(model_path),
            framework_type=FrameworkType.ONNX,
            framework_metadata={
                "model_path": model_path,
                "simulation_mode": True,
                "error": "ONNX not available",
            },
        )

        # Create a simple placeholder model
        input_tensor = TensorInfo(
            name="input",
            shape=TensorShape([1, 3, 224, 224]),  # ONNX typically uses NCHW format
            dtype=DataType.FLOAT32,
            framework_metadata={"simulated": True},
        )

        output_tensor = TensorInfo(
            name="output",
            shape=TensorShape([1, 1000]),
            dtype=DataType.FLOAT32,
            framework_metadata={"simulated": True},
        )

        model_node = UIRNode(
            node_id="model",
            name="Simulated ONNX Model",
            operation_type=OperationType.CUSTOM,
            framework_type=FrameworkType.ONNX,
            framework_metadata={"simulated": True},
        )

        graph.add_tensor(input_tensor)
        graph.add_tensor(output_tensor)
        graph.add_node(model_node)
        graph.add_edge("input", "model", "input")
        graph.add_edge("model", "output", "output")

        return graph

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return [".onnx"]

    def get_framework_type(self) -> FrameworkType:
        """Get the framework type."""
        return FrameworkType.ONNX


class PyTorchParser(FrameworkParser):
    """Parser for PyTorch models."""

    def __init__(self):
        self.torch_available = False
        try:
            import torch

            self.torch = torch
            self.torch_available = True
            logger.info("PyTorch parser initialized successfully")
        except ImportError:
            logger.warning("PyTorch not available, parser will use simulation mode")

    def parse_model(self, model_path: str) -> UIRGraph:
        """Parse a PyTorch model into UIR."""
        if not self.torch_available:
            return self._simulate_parsing(model_path)

        try:
            # Load PyTorch model
            model = self.torch.load(model_path, map_location="cpu")

            graph = UIRGraph(
                name=os.path.basename(model_path),
                framework_type=FrameworkType.PYTORCH,
                framework_metadata={
                    "model_path": model_path,
                    "model_type": "pytorch",
                    "torch_version": self.torch.__version__,
                },
            )

            # For now, create a simplified representation
            # In a full implementation, we would traverse the model's modules
            model_node = UIRNode(
                node_id="pytorch_model",
                name="PyTorch Model",
                operation_type=OperationType.CUSTOM,
                framework_type=FrameworkType.PYTORCH,
                framework_metadata={
                    "model_state_dict": True,
                    "model_type": type(model).__name__,
                },
            )

            # Add placeholder input and output tensors
            input_tensor = TensorInfo(
                name="input",
                shape=TensorShape([1, 3, 224, 224]),  # PyTorch uses NCHW format
                dtype=DataType.FLOAT32,
                framework_metadata={"pytorch_input": True},
            )

            output_tensor = TensorInfo(
                name="output",
                shape=TensorShape([1, 1000]),
                dtype=DataType.FLOAT32,
                framework_metadata={"pytorch_output": True},
            )

            graph.add_tensor(input_tensor)
            graph.add_tensor(output_tensor)
            graph.add_node(model_node)
            graph.add_edge("input", "pytorch_model", "input")
            graph.add_edge("pytorch_model", "output", "output")

            return graph

        except Exception as e:
            logger.error(f"Failed to parse PyTorch model {model_path}: {e}")
            return self._simulate_parsing(model_path)

    def _simulate_parsing(self, model_path: str) -> UIRGraph:
        """Simulate parsing when PyTorch is not available."""
        logger.warning(f"Simulating PyTorch model parsing for {model_path}")

        graph = UIRGraph(
            name=os.path.basename(model_path),
            framework_type=FrameworkType.PYTORCH,
            framework_metadata={
                "model_path": model_path,
                "simulation_mode": True,
                "error": "PyTorch not available",
            },
        )

        # Create a simple placeholder model
        input_tensor = TensorInfo(
            name="input",
            shape=TensorShape([1, 3, 224, 224]),
            dtype=DataType.FLOAT32,
            framework_metadata={"simulated": True},
        )

        output_tensor = TensorInfo(
            name="output",
            shape=TensorShape([1, 1000]),
            dtype=DataType.FLOAT32,
            framework_metadata={"simulated": True},
        )

        model_node = UIRNode(
            node_id="model",
            name="Simulated PyTorch Model",
            operation_type=OperationType.CUSTOM,
            framework_type=FrameworkType.PYTORCH,
            framework_metadata={"simulated": True},
        )

        graph.add_tensor(input_tensor)
        graph.add_tensor(output_tensor)
        graph.add_node(model_node)
        graph.add_edge("input", "model", "input")
        graph.add_edge("model", "output", "output")

        return graph

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return [".pth", ".pt", ".pkl"]

    def get_framework_type(self) -> FrameworkType:
        """Get the framework type."""
        return FrameworkType.PYTORCH


class FrameworkParserRegistry:
    """Registry for framework parsers."""

    def __init__(self):
        self.parsers: Dict[FrameworkType, FrameworkParser] = {}
        self._register_default_parsers()

    def _register_default_parsers(self):
        """Register default framework parsers."""
        self.register_parser(TensorFlowParser())
        self.register_parser(ONNXParser())
        self.register_parser(PyTorchParser())

    def register_parser(self, parser: FrameworkParser):
        """Register a framework parser."""
        self.parsers[parser.get_framework_type()] = parser
        logger.info(f"Registered parser for {parser.get_framework_type().value}")

    def get_parser(self, framework_type: FrameworkType) -> Optional[FrameworkParser]:
        """Get a parser for a specific framework."""
        return self.parsers.get(framework_type)

    def parse_model(self, model_path: str) -> UIRGraph:
        """Parse a model using the appropriate parser based on file extension."""
        model_path = Path(model_path)
        extension = model_path.suffix.lower()

        # Determine framework type from file extension
        if extension in [".h5", ".keras", ".pb"] or model_path.is_dir():
            framework_type = FrameworkType.TENSORFLOW
        elif extension == ".onnx":
            framework_type = FrameworkType.ONNX
        elif extension in [".pth", ".pt", ".pkl"]:
            framework_type = FrameworkType.PYTORCH
        elif extension == ".tflite":
            framework_type = FrameworkType.TFLITE
        else:
            raise ValueError(f"Unsupported model format: {extension}")

        parser = self.get_parser(framework_type)
        if not parser:
            raise ValueError(
                f"No parser available for framework: {framework_type.value}"
            )

        return parser.parse_model(str(model_path))

    def get_supported_formats(self) -> List[str]:
        """Get all supported file formats."""
        formats = []
        for parser in self.parsers.values():
            formats.extend(parser.get_supported_formats())
        return list(set(formats))


# Global registry instance
parser_registry = FrameworkParserRegistry()


def parse_model_to_uir(model_path: str) -> UIRGraph:
    """Parse a model from any supported framework into UIR.

    Args:
        model_path: Path to the model file

    Returns:
        UIRGraph: Unified intermediate representation of the model

    Raises:
        ValueError: If the model format is not supported
    """
    return parser_registry.parse_model(model_path)


if __name__ == "__main__":
    # Test the framework parsers
    test_models = [
        "test_model.tflite",
        "test_model.onnx",
        "test_model.pth",
        "test_model.h5",
    ]

    for model_path in test_models:
        try:
            print(f"\n=== Testing {model_path} ===")
            graph = parse_model_to_uir(model_path)
            print(f"  Framework: {graph.framework_type.value}")
            print(f"  Nodes: {len(graph.nodes)}")
            print(f"  Tensors: {len(graph.tensors)}")
            print(f"  Edges: {len(graph.edges)}")

            # Validate the graph
            is_valid, errors = graph.validate_graph()
            print(f"  Valid: {is_valid}")
            if errors:
                print(f"  Errors: {errors}")

        except Exception as e:
            print(f"  Error: {e}")
