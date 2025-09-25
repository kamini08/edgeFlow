"""Abstract Syntax Tree (AST) for EdgeFlow DSL.

This module defines the AST nodes that represent the syntactic structure
of EdgeFlow configuration files. The AST serves as an intermediate
representation between parsing and code generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class ASTNode(ABC):
    """Base class for all AST nodes."""

    @abstractmethod
    def accept(self, visitor: "ASTVisitor") -> Any:
        """Accept a visitor for the visitor pattern."""
        pass


class Statement(ASTNode):
    """Base class for all statement types."""

    pass


class Expression(ASTNode):
    """Base class for all expression types."""

    pass


# ============================================================================
# Type Constraint Enums
# ============================================================================


class LayerType(Enum):
    """Enumeration of supported layer types with validation."""

    CONV2D = "Conv2D"
    CONV1D = "Conv1D"
    DENSE = "Dense"
    MAXPOOL2D = "MaxPool2D"
    AVGPOOL2D = "AvgPool2D"
    FLATTEN = "Flatten"
    DROPOUT = "Dropout"
    BATCH_NORM = "BatchNorm"
    LAYER_NORM = "LayerNorm"
    ACTIVATION = "Activation"
    LSTM = "LSTM"
    GRU = "GRU"
    EMBEDDING = "Embedding"
    ATTENTION = "Attention"


class ActivationType(Enum):
    """Enumeration of supported activation functions."""

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LEAKY_RELU = "leaky_relu"
    SWISH = "swish"
    GELU = "gelu"
    LINEAR = "linear"


class PaddingType(Enum):
    """Enumeration of supported padding types."""

    VALID = "valid"
    SAME = "same"


class DataType(Enum):
    """Enumeration of supported data types."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"


# ============================================================================
# Type-Constrained Value Classes
# ============================================================================


@dataclass
class ConstrainedInt:
    """Integer with validation constraints."""

    value: int
    min_val: int = 1
    max_val: int = 10000

    def __post_init__(self):
        if not (self.min_val <= self.value <= self.max_val):
            raise ValueError(
                f"Value {self.value} must be between {self.min_val} and {self.max_val}"
            )


@dataclass
class KernelSize:
    """Kernel size with validation (1-15 for most layers)."""

    size: Union[int, Tuple[int, int]]

    def __post_init__(self):
        if isinstance(self.size, int):
            if not (1 <= self.size <= 15):
                raise ValueError(f"Kernel size {self.size} must be between 1 and 15")
        elif isinstance(self.size, tuple):
            if len(self.size) != 2:
                raise ValueError("Kernel size tuple must have exactly 2 elements")
            for s in self.size:
                if not (1 <= s <= 15):
                    raise ValueError(f"Kernel size {s} must be between 1 and 15")


@dataclass
class StrideValue:
    """Stride value with validation (1-8)."""

    stride: Union[int, Tuple[int, int]]

    def __post_init__(self):
        if isinstance(self.stride, int):
            if not (1 <= self.stride <= 8):
                raise ValueError(f"Stride {self.stride} must be between 1 and 8")
        elif isinstance(self.stride, tuple):
            if len(self.stride) != 2:
                raise ValueError("Stride tuple must have exactly 2 elements")
            for s in self.stride:
                if not (1 <= s <= 8):
                    raise ValueError(f"Stride {s} must be between 1 and 8")


@dataclass
class DropoutRate:
    """Dropout rate with validation (0.0-0.9)."""

    rate: float

    def __post_init__(self):
        if not (0.0 <= self.rate <= 0.9):
            raise ValueError(f"Dropout rate {self.rate} must be between 0.0 and 0.9")


# ============================================================================
# Statement Nodes
# ============================================================================


@dataclass
class ModelStatement(Statement):
    """Represents a model path statement: model: "path/to/model.tflite" """

    path: str

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_model_statement(self)


@dataclass
class QuantizeStatement(Statement):
    """Represents a quantization statement: quantize: int8"""

    quant_type: str  # 'int8', 'float16', 'dynamic', 'none'

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_quantize_statement(self)


@dataclass
class TargetDeviceStatement(Statement):
    """Represents a target device statement: target_device: raspberry_pi"""

    device: str

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_target_device_statement(self)


@dataclass
class DeployPathStatement(Statement):
    """Represents a deployment path statement: deploy_path: "/models/" """

    path: str

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_deploy_path_statement(self)


@dataclass
class InputStreamStatement(Statement):
    """Represents an input stream statement: input_stream: camera"""

    stream: str

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_input_stream_statement(self)


@dataclass
class BufferSizeStatement(Statement):
    """Represents a buffer size statement: buffer_size: 32"""

    size: int

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_buffer_size_statement(self)


@dataclass
class OptimizeForStatement(Statement):
    """Represents an optimization goal statement: optimize_for: latency"""

    goal: str  # 'latency', 'memory', 'accuracy', 'power'

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_optimize_for_statement(self)


@dataclass
class MemoryLimitStatement(Statement):
    """Represents a memory limit statement: memory_limit: 64 MB"""

    limit_mb: ConstrainedInt = field(default_factory=lambda: ConstrainedInt(64, 16, 10000))

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_memory_limit_statement(self)


@dataclass
class FusionStatement(Statement):
    """Represents a fusion statement: enable_fusion: true"""

    enabled: bool

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_fusion_statement(self)


@dataclass
class FrameworkStatement(Statement):
    """Represents a framework specification: framework = "pytorch" """

    framework: str

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_framework_statement(self)


@dataclass
class HybridOptimizationStatement(Statement):
    """Represents hybrid optimization enablement: enable_hybrid_optimization = true """

    enabled: bool

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_hybrid_optimization_statement(self)


@dataclass
class PyTorchQuantizeStatement(Statement):
    """Represents PyTorch-specific quantization: pytorch_quantize = "dynamic_int8" """

    quantize_type: str

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_pytorch_quantize_statement(self)


@dataclass
class FineTuningStatement(Statement):
    """Represents fine-tuning configuration: fine_tuning = true """

    enabled: bool

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_fine_tuning_statement(self)


@dataclass
class ConditionalStatement(Statement):
    """Represents a conditional statement: if condition then statements end"""

    condition: "Condition"
    then_block: List[Statement]
    else_block: Optional[List[Statement]] = None

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_conditional_statement(self)


@dataclass
class PipelineStatement(Statement):
    """Represents a pipeline statement: pipeline: { preprocess, inference, postprocess }"""

    steps: List[str]

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_pipeline_statement(self)


# ============================================================================
# Layer Declaration Nodes with Type Constraints
# ============================================================================


@dataclass
class LayerDeclaration(Statement):
    """Base class for layer declarations with type validation."""

    name: str = ""
    layer_type: LayerType = LayerType.CONV2D
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_line: Optional[int] = None
    source_column: Optional[int] = None

    def __post_init__(self):
        """Post-initialization hook for subclasses."""
        pass

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_layer_declaration(self)

    def validate_parameters(self) -> List[str]:
        """Validate layer parameters and return list of errors."""
        return []  # Override in subclasses


@dataclass
class Conv2DDeclaration(LayerDeclaration):
    """Conv2D layer with type-constrained parameters."""

    filters: ConstrainedInt = field(default_factory=lambda: ConstrainedInt(32))
    kernel_size: KernelSize = field(default_factory=lambda: KernelSize(3))
    strides: StrideValue = field(default_factory=lambda: StrideValue(1))
    padding: PaddingType = PaddingType.VALID
    activation: ActivationType = ActivationType.LINEAR
    use_bias: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.layer_type = LayerType.CONV2D
        self.parameters = {
            "filters": self.filters.value,
            "kernel_size": self.kernel_size.size,
            "strides": self.strides.stride,
            "padding": self.padding.value,
            "activation": self.activation.value,
            "use_bias": self.use_bias,
        }

    def validate_parameters(self) -> List[str]:
        """Validate Conv2D-specific parameters."""
        errors = []

        # Additional validation beyond type constraints
        if self.filters.value > 2048:
            errors.append(
                f"Conv2D filters {self.filters.value} exceeds recommended maximum (2048)"
            )

        if isinstance(self.kernel_size.size, int) and self.kernel_size.size > 11:
            errors.append(
                f"Conv2D kernel size {self.kernel_size.size} may be too large for edge devices"
            )
        elif isinstance(self.kernel_size.size, tuple):
            for s in self.kernel_size.size:
                if s > 11:
                    errors.append(
                        f"Conv2D kernel size {s} may be too large for edge devices"
                    )

        return errors


@dataclass
class DenseDeclaration(LayerDeclaration):
    """Dense layer with type-constrained parameters."""

    units: ConstrainedInt = field(default_factory=lambda: ConstrainedInt(128))
    activation: ActivationType = ActivationType.LINEAR
    use_bias: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.layer_type = LayerType.DENSE
        self.parameters = {
            "units": self.units.value,
            "activation": self.activation.value,
            "use_bias": self.use_bias,
        }

    def validate_parameters(self) -> List[str]:
        """Validate Dense-specific parameters."""
        errors = []

        if self.units.value > 4096:
            errors.append(
                f"Dense units {self.units.value} may exceed memory limits on edge devices"
            )

        return errors


@dataclass
class DropoutDeclaration(LayerDeclaration):
    """Dropout layer with type-constrained parameters."""

    rate: DropoutRate = field(default_factory=lambda: DropoutRate(0.5))

    def __post_init__(self):
        super().__post_init__()
        self.layer_type = LayerType.DROPOUT
        self.parameters = {"rate": self.rate.rate}

    def validate_parameters(self) -> List[str]:
        """Validate Dropout-specific parameters."""
        errors = []

        if self.rate.rate > 0.7:
            errors.append(
                f"Dropout rate {self.rate.rate} is very high and may hurt model performance"
            )

        return errors


@dataclass
class MaxPool2DDeclaration(LayerDeclaration):
    """MaxPool2D layer with type-constrained parameters."""

    pool_size: Union[int, Tuple[int, int]] = 2
    strides: Optional[StrideValue] = None
    padding: PaddingType = PaddingType.VALID

    def __post_init__(self):
        super().__post_init__()
        self.layer_type = LayerType.MAXPOOL2D

        # Validate pool_size
        if isinstance(self.pool_size, int):
            if not (1 <= self.pool_size <= 8):
                raise ValueError(f"Pool size {self.pool_size} must be between 1 and 8")
        elif isinstance(self.pool_size, tuple):
            for s in self.pool_size:
                if not (1 <= s <= 8):
                    raise ValueError(f"Pool size {s} must be between 1 and 8")

        self.parameters = {
            "pool_size": self.pool_size,
            "strides": self.strides.stride if self.strides else self.pool_size,
            "padding": self.padding.value,
        }


# ============================================================================
# Expression Nodes
# ============================================================================


@dataclass
class Literal(Expression):
    """Represents a literal value (string, number, boolean)."""

    value: Union[str, int, float, bool]

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_literal(self)


@dataclass
class Identifier(Expression):
    """Represents an identifier/variable name."""

    name: str

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_identifier(self)


@dataclass
class BinaryExpression(Expression):
    """Represents a binary expression: left operator right"""

    left: Expression
    operator: str
    right: Expression

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_binary_expression(self)


@dataclass
class UnaryExpression(Expression):
    """Represents a unary expression: operator operand"""

    operator: str
    operand: Expression

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_unary_expression(self)


# ============================================================================
# Condition Nodes
# ============================================================================


@dataclass
class Condition(ASTNode):
    """Represents a condition in an if statement."""

    left: Expression
    operator: str  # '==', '!=', '<', '>', '<=', '>='
    right: Expression

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_condition(self)


# ============================================================================
# Program Node
# ============================================================================


@dataclass
class Program(ASTNode):
    """Represents the root of the AST - a complete EdgeFlow program."""

    statements: List[Statement]

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_program(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the program to a dictionary representation."""
        return {
            "type": "Program",
            "statements": [
                {"type": type(stmt).__name__, "data": stmt.__dict__}
                for stmt in self.statements
            ],
        }


# ============================================================================
# Visitor Pattern
# ============================================================================


class ASTVisitor(ABC):
    """Base visitor class for AST traversal."""

    @abstractmethod
    def visit_program(self, node: Program) -> Any:
        pass

    @abstractmethod
    def visit_model_statement(self, node: ModelStatement) -> Any:
        pass

    @abstractmethod
    def visit_quantize_statement(self, node: QuantizeStatement) -> Any:
        pass

    @abstractmethod
    def visit_target_device_statement(self, node: TargetDeviceStatement) -> Any:
        pass

    @abstractmethod
    def visit_deploy_path_statement(self, node: DeployPathStatement) -> Any:
        pass

    @abstractmethod
    def visit_input_stream_statement(self, node: InputStreamStatement) -> Any:
        pass

    @abstractmethod
    def visit_buffer_size_statement(self, node: BufferSizeStatement) -> Any:
        pass

    @abstractmethod
    def visit_optimize_for_statement(self, node: OptimizeForStatement) -> Any:
        pass

    @abstractmethod
    def visit_memory_limit_statement(self, node: MemoryLimitStatement) -> Any:
        pass

    @abstractmethod
    def visit_fusion_statement(self, node: FusionStatement) -> Any:
        pass

    @abstractmethod
    def visit_framework_statement(self, node: FrameworkStatement) -> Any:
        pass

    @abstractmethod
    def visit_hybrid_optimization_statement(self, node: HybridOptimizationStatement) -> Any:
        pass

    @abstractmethod
    def visit_pytorch_quantize_statement(self, node: PyTorchQuantizeStatement) -> Any:
        pass

    @abstractmethod
    def visit_fine_tuning_statement(self, node: FineTuningStatement) -> Any:
        pass

    @abstractmethod
    def visit_conditional_statement(self, node: ConditionalStatement) -> Any:
        pass

    @abstractmethod
    def visit_pipeline_statement(self, node: PipelineStatement) -> Any:
        pass

    @abstractmethod
    def visit_literal(self, node: Literal) -> Any:
        pass

    @abstractmethod
    def visit_identifier(self, node: Identifier) -> Any:
        pass

    @abstractmethod
    def visit_binary_expression(self, node: BinaryExpression) -> Any:
        pass

    @abstractmethod
    def visit_unary_expression(self, node: UnaryExpression) -> Any:
        pass

    @abstractmethod
    def visit_condition(self, node: Condition) -> Any:
        pass

    @abstractmethod
    def visit_layer_declaration(self, node: LayerDeclaration) -> Any:
        pass


# ============================================================================
# Utility Functions
# ============================================================================


def create_program_from_dict(config: Dict[str, Any]) -> Program:
    """Create an AST Program from a parsed configuration dictionary.

    This is a helper function to convert the current parser output
    into a proper AST structure.

    Args:
        config: Parsed configuration dictionary from the parser

    Returns:
        Program: AST representation of the configuration
    """
    statements: List[Statement] = []

    # Convert dictionary entries to AST statements
    if "model" in config:
        statements.append(ModelStatement(path=config["model"]))

    if "quantize" in config:
        statements.append(QuantizeStatement(quant_type=config["quantize"]))

    if "target_device" in config:
        statements.append(TargetDeviceStatement(device=config["target_device"]))

    if "deploy_path" in config:
        statements.append(DeployPathStatement(path=config["deploy_path"]))

    if "input_stream" in config:
        statements.append(InputStreamStatement(stream=config["input_stream"]))

    if "buffer_size" in config:
        statements.append(BufferSizeStatement(size=config["buffer_size"]))

    if "optimize_for" in config:
        statements.append(OptimizeForStatement(goal=config["optimize_for"]))

    if "memory_limit" in config:
        statements.append(MemoryLimitStatement(limit_mb=ConstrainedInt(config["memory_limit"], 16, 10000)))

    if "enable_fusion" in config:
        statements.append(FusionStatement(enabled=config["enable_fusion"]))

    if "framework" in config:
        statements.append(FrameworkStatement(framework=config["framework"]))

    if "enable_hybrid_optimization" in config:
        statements.append(HybridOptimizationStatement(enabled=config["enable_hybrid_optimization"]))

    if "pytorch_quantize" in config:
        statements.append(PyTorchQuantizeStatement(quantize_type=config["pytorch_quantize"]))

    if "fine_tuning" in config:
        statements.append(FineTuningStatement(enabled=config["fine_tuning"]))

    return Program(statements=statements)


def print_ast(node: ASTNode, indent: int = 0) -> str:
    """Pretty print an AST node for debugging.

    Args:
        node: AST node to print
        indent: Current indentation level

    Returns:
        str: Pretty-printed AST representation
    """
    prefix = "  " * indent
    result = []

    if isinstance(node, Program):
        result.append(f"{prefix}Program:")
        for stmt in node.statements:
            result.append(print_ast(stmt, indent + 1))
    elif isinstance(node, ModelStatement):
        result.append(f"{prefix}ModelStatement(path='{node.path}')")
    elif isinstance(node, QuantizeStatement):
        result.append(f"{prefix}QuantizeStatement(quant_type='{node.quant_type}')")
    elif isinstance(node, TargetDeviceStatement):
        result.append(f"{prefix}TargetDeviceStatement(device='{node.device}')")
    elif isinstance(node, DeployPathStatement):
        result.append(f"{prefix}DeployPathStatement(path='{node.path}')")
    elif isinstance(node, InputStreamStatement):
        result.append(f"{prefix}InputStreamStatement(stream='{node.stream}')")
    elif isinstance(node, BufferSizeStatement):
        result.append(f"{prefix}BufferSizeStatement(size={node.size})")
    elif isinstance(node, OptimizeForStatement):
        result.append(f"{prefix}OptimizeForStatement(goal='{node.goal}')")
    elif isinstance(node, MemoryLimitStatement):
        result.append(f"{prefix}MemoryLimitStatement(limit_mb={node.limit_mb.value})")
    elif isinstance(node, FusionStatement):
        result.append(f"{prefix}FusionStatement(enabled={node.enabled})")
    elif isinstance(node, FrameworkStatement):
        result.append(f"{prefix}FrameworkStatement(framework='{node.framework}')")
    elif isinstance(node, HybridOptimizationStatement):
        result.append(f"{prefix}HybridOptimizationStatement(enabled={node.enabled})")
    elif isinstance(node, PyTorchQuantizeStatement):
        result.append(f"{prefix}PyTorchQuantizeStatement(quantize_type='{node.quantize_type}')")
    elif isinstance(node, FineTuningStatement):
        result.append(f"{prefix}FineTuningStatement(enabled={node.enabled})")
    elif isinstance(node, ConditionalStatement):
        result.append(f"{prefix}ConditionalStatement:")
        result.append(f"{prefix}  condition: {print_ast(node.condition, indent + 2)}")
        result.append(f"{prefix}  then_block:")
        for stmt in node.then_block:
            result.append(print_ast(stmt, indent + 2))
        if node.else_block:
            result.append(f"{prefix}  else_block:")
            for stmt in node.else_block:
                result.append(print_ast(stmt, indent + 2))
    elif isinstance(node, PipelineStatement):
        result.append(f"{prefix}PipelineStatement(steps={node.steps})")
    elif isinstance(node, Literal):
        result.append(f"{prefix}Literal(value={repr(node.value)})")
    elif isinstance(node, Identifier):
        result.append(f"{prefix}Identifier(name='{node.name}')")
    elif isinstance(node, BinaryExpression):
        result.append(f"{prefix}BinaryExpression(operator='{node.operator}'):")
        result.append(print_ast(node.left, indent + 1))
        result.append(print_ast(node.right, indent + 1))
    elif isinstance(node, UnaryExpression):
        result.append(f"{prefix}UnaryExpression(operator='{node.operator}'):")
        result.append(print_ast(node.operand, indent + 1))
    elif isinstance(node, Condition):
        result.append(f"{prefix}Condition(operator='{node.operator}'):")
        result.append(print_ast(node.left, indent + 1))
        result.append(print_ast(node.right, indent + 1))
    else:
        result.append(f"{prefix}UnknownNode({type(node).__name__})")

    return "\n".join(result)
