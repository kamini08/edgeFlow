"""
Semantic Analyzer for DSL Compiler - Main Package

This package provides comprehensive semantic analysis for DSL intermediate representations,
including shape validation, parameter checking, resource constraints, and device compatibility.
"""

from .analyzer import SemanticAnalyzer, semantic_check
from .constraints import (
    ConstraintConfig,
    DeviceConstraints,
    ParameterRange,
    get_edge_device_config,
    get_mobile_device_config,
    get_server_device_config,
)
from .error_types import (
    ErrorCollector,
    ErrorSeverity,
    ErrorType,
    SemanticError,
    SourceLocation,
)
from .ir_nodes import (
    ActivationType,
    DataType,
    IRGraph,
    IRNode,
    LayerType,
    TensorInfo,
    TensorShape,
    create_conv2d_node,
    create_dense_node,
    create_input_node,
)

__version__ = "1.0.0"
__author__ = "EdgeFlow Team"

__all__ = [
    # Error handling
    "ErrorSeverity",
    "ErrorType",
    "SourceLocation",
    "SemanticError",
    "ErrorCollector",
    # IR structures
    "DataType",
    "ActivationType",
    "TensorShape",
    "TensorInfo",
    "LayerType",
    "IRNode",
    "IRGraph",
    "create_input_node",
    "create_dense_node",
    "create_conv2d_node",
    # Constraints and configuration
    "ParameterRange",
    "DeviceConstraints",
    "ConstraintConfig",
    "get_edge_device_config",
    "get_mobile_device_config",
    "get_server_device_config",
    # Main analyzer
    "SemanticAnalyzer",
    "semantic_check",
]
