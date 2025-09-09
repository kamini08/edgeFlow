"""Abstract Syntax Tree (AST) for EdgeFlow DSL.

This module defines the AST nodes that represent the syntactic structure
of EdgeFlow configuration files. The AST serves as an intermediate
representation between parsing and code generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


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

    limit_mb: int

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_memory_limit_statement(self)


@dataclass
class FusionStatement(Statement):
    """Represents a fusion statement: enable_fusion: true"""

    enabled: bool

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_fusion_statement(self)


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
        statements.append(MemoryLimitStatement(limit_mb=config["memory_limit"]))

    if "enable_fusion" in config:
        statements.append(FusionStatement(enabled=config["enable_fusion"]))

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
        result.append(f"{prefix}MemoryLimitStatement(limit_mb={node.limit_mb})")
    elif isinstance(node, FusionStatement):
        result.append(f"{prefix}FusionStatement(enabled={node.enabled})")
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
