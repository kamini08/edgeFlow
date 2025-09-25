"""
Grammar-aware semantic validator that integrates with ANTLR parser and existing semantic analyzer.
This module provides early validation at the grammar/AST level before IR conversion.
"""
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgeflow_ast import (
    ActivationType,
    ASTNode,
    ASTVisitor,
    ConstrainedInt,
    Conv2DDeclaration,
    DataType,
    DenseDeclaration,
    DropoutDeclaration,
    DropoutRate,
    KernelSize,
    LayerDeclaration,
    LayerType,
    MaxPool2DDeclaration,
    PaddingType,
    Program,
    StrideValue,
)
from semantic_analyzer.constraints import ConstraintConfig, DeviceConstraints
from semantic_analyzer.error_types import (
    ErrorCollector,
    ErrorSeverity,
    ErrorType,
    SemanticError,
    SourceLocation,
)


@dataclass
class GrammarValidationError:
    """Grammar-level validation error with source location."""

    message: str
    error_type: str
    line: Optional[int] = None
    column: Optional[int] = None
    layer_name: Optional[str] = None
    suggestion: Optional[str] = None


class GrammarSemanticValidator(ASTVisitor):
    """
    Grammar-aware semantic validator that performs early validation
    during AST traversal, before IR conversion.
    """

    def __init__(self, device_config: Optional[ConstraintConfig] = None):
        self.device_config = device_config or ConstraintConfig()
        self.errors: List[GrammarValidationError] = []
        self.warnings: List[GrammarValidationError] = []
        self.layer_names: Dict[str, LayerType] = {}
        self.layer_sequence: List[Tuple[str, LayerType]] = []

    def validate_ast(self, ast: Program) -> ErrorCollector:
        """
        Validate the entire AST and return structured errors.

        Args:
            ast: The parsed AST to validate

        Returns:
            ErrorCollector with validation results
        """
        self.errors.clear()
        self.warnings.clear()
        self.layer_names.clear()
        self.layer_sequence.clear()

        # Visit the AST
        ast.accept(self)

        # Convert to ErrorCollector format
        error_collector = ErrorCollector()

        for error in self.errors:
            location = (
                SourceLocation(line=error.line or 0, column=error.column or 0)
                if error.line
                else None
            )

            semantic_error = SemanticError(
                error_type=self._map_error_type(error.error_type),
                severity=ErrorSeverity.ERROR,
                message=error.message,
                layer_name=error.layer_name,
                location=location,
                suggestion=error.suggestion,
            )
            error_collector.add_error(semantic_error)

        for warning in self.warnings:
            location = (
                SourceLocation(line=warning.line or 0, column=warning.column or 0)
                if warning.line
                else None
            )

            semantic_error = SemanticError(
                error_type=self._map_error_type(warning.error_type),
                severity=ErrorSeverity.WARNING,
                message=warning.message,
                layer_name=warning.layer_name,
                location=location,
                suggestion=warning.suggestion,
            )
            error_collector.add_error(semantic_error)

        return error_collector

    def _map_error_type(self, error_type: str) -> ErrorType:
        """Map grammar error types to semantic error types."""
        mapping = {
            "parameter_range": ErrorType.PARAM_RANGE,
            "forbidden_sequence": ErrorType.FORBIDDEN_CONFIG,
            "device_compatibility": ErrorType.DEVICE_INCOMPATIBLE,
            "resource_limit": ErrorType.RESOURCE_LIMIT,
            "invalid_value": ErrorType.PARAM_RANGE,
            "type_mismatch": ErrorType.DATATYPE_MISMATCH,
        }
        return mapping.get(error_type, ErrorType.PARAM_RANGE)

    def _add_error(
        self,
        message: str,
        error_type: str = "parameter_range",
        line: Optional[int] = None,
        column: Optional[int] = None,
        layer_name: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """Add a validation error."""
        self.errors.append(
            GrammarValidationError(
                message=message,
                error_type=error_type,
                line=line,
                column=column,
                layer_name=layer_name,
                suggestion=suggestion,
            )
        )

    def _add_warning(
        self,
        message: str,
        error_type: str = "parameter_range",
        line: Optional[int] = None,
        column: Optional[int] = None,
        layer_name: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """Add a validation warning."""
        self.warnings.append(
            GrammarValidationError(
                message=message,
                error_type=error_type,
                line=line,
                column=column,
                layer_name=layer_name,
                suggestion=suggestion,
            )
        )

    # ========================================================================
    # AST Visitor Methods
    # ========================================================================

    def visit_program(self, node: Program) -> Any:
        """Validate the entire program."""
        # Visit all statements
        for statement in node.statements:
            statement.accept(self)

        # Perform cross-layer validation
        self._validate_layer_sequence()
        self._validate_device_compatibility()

        return None

    def visit_layer_declaration(self, node: LayerDeclaration) -> Any:
        """Validate a generic layer declaration."""
        # Check for duplicate layer names
        if node.name in self.layer_names:
            self._add_error(
                f"Duplicate layer name '{node.name}'",
                error_type="invalid_value",
                line=node.source_line,
                column=node.source_column,
                layer_name=node.name,
                suggestion="Use unique names for all layers",
            )
        else:
            self.layer_names[node.name] = node.layer_type
            self.layer_sequence.append((node.name, node.layer_type))

        # Validate layer-specific parameters
        param_errors = node.validate_parameters()
        for error_msg in param_errors:
            self._add_warning(
                error_msg,
                error_type="parameter_range",
                line=node.source_line,
                column=node.source_column,
                layer_name=node.name,
            )

        return None

    def _validate_layer_sequence(self):
        """Validate the sequence of layers for forbidden combinations."""
        for i in range(len(self.layer_sequence) - 1):
            current_name, current_type = self.layer_sequence[i]
            next_name, next_type = self.layer_sequence[i + 1]

            # Check forbidden sequences
            if self._is_forbidden_sequence(current_type, next_type):
                self._add_error(
                    f"Forbidden layer sequence: {current_type.value} -> {next_type.value}",
                    error_type="forbidden_sequence",
                    layer_name=next_name,
                    suggestion=self._get_sequence_suggestion(current_type, next_type),
                )

    def _is_forbidden_sequence(
        self, prev_type: LayerType, curr_type: LayerType
    ) -> bool:
        """Check if a layer sequence is forbidden."""
        forbidden_sequences = [
            (LayerType.CONV2D, LayerType.DENSE),
            (LayerType.MAXPOOL2D, LayerType.DENSE),
            (LayerType.AVGPOOL2D, LayerType.DENSE),
        ]
        return (prev_type, curr_type) in forbidden_sequences

    def _get_sequence_suggestion(
        self, prev_type: LayerType, curr_type: LayerType
    ) -> str:
        """Get suggestion for fixing forbidden sequences."""
        if (
            prev_type in [LayerType.CONV2D, LayerType.MAXPOOL2D, LayerType.AVGPOOL2D]
            and curr_type == LayerType.DENSE
        ):
            return (
                "Add a Flatten layer between the convolutional/pooling and dense layers"
            )
        return "Review layer sequence and add appropriate intermediate layers"

    def _validate_device_compatibility(self):
        """Validate compatibility with target device constraints."""
        device_limits = self.device_config.device_constraints

        for layer_name, layer_type in self.layer_names.items():
            # Check if layer type is supported
            if layer_type not in device_limits.supported_layers:
                self._add_error(
                    f"Layer type {layer_type.value} is not supported on target device",
                    error_type="device_compatibility",
                    layer_name=layer_name,
                    suggestion=f"Use supported layer types: {[lt.value for lt in device_limits.supported_layers]}",
                )

    # ========================================================================
    # Specific Layer Validation Methods
    # ========================================================================

    def validate_conv2d_parameters(self, layer: Conv2DDeclaration) -> List[str]:
        """Validate Conv2D parameters against device constraints."""
        errors = []
        device_limits = self.device_config.device_constraints

        # Check filter count
        if layer.filters.value > device_limits.max_filters:
            errors.append(
                f"Conv2D filters {layer.filters.value} exceeds device limit {device_limits.max_filters}"
            )

        # Check kernel size
        max_kernel = layer.kernel_size.size
        if isinstance(max_kernel, tuple):
            max_kernel = max(max_kernel)

        if max_kernel > device_limits.max_kernel_size:
            errors.append(
                f"Conv2D kernel size {max_kernel} exceeds device limit {device_limits.max_kernel_size}"
            )

        return errors

    def validate_dense_parameters(self, layer: DenseDeclaration) -> List[str]:
        """Validate Dense parameters against device constraints."""
        errors = []
        device_limits = self.device_config.device_constraints

        # Check unit count
        if layer.units.value > device_limits.max_units:
            errors.append(
                f"Dense units {layer.units.value} exceeds device limit {device_limits.max_units}"
            )

        return errors

    # ========================================================================
    # Required visitor methods (delegating to base implementation)
    # ========================================================================

    def visit_model_statement(self, node) -> Any:
        return None

    def visit_quantize_statement(self, node) -> Any:
        return None

    def visit_target_device_statement(self, node) -> Any:
        return None

    def visit_deploy_path_statement(self, node) -> Any:
        return None

    def visit_input_stream_statement(self, node) -> Any:
        return None

    def visit_buffer_size_statement(self, node) -> Any:
        return None

    def visit_optimize_for_statement(self, node) -> Any:
        return None

    def visit_memory_limit_statement(self, node) -> Any:
        return None

    def visit_fusion_statement(self, node) -> Any:
        return None

    def visit_conditional_statement(self, node) -> Any:
        return None

    def visit_pipeline_statement(self, node) -> Any:
        return None

    def visit_literal(self, node) -> Any:
        return None

    def visit_identifier(self, node) -> Any:
        return None

    def visit_binary_expression(self, node) -> Any:
        return None

    def visit_unary_expression(self, node) -> Any:
        return None

    def visit_condition(self, node) -> Any:
        return None


# ============================================================================
# Factory Functions for Creating Type-Constrained Layers
# ============================================================================


def create_conv2d_from_params(
    name: str,
    params: Dict[str, Any],
    line: Optional[int] = None,
    column: Optional[int] = None,
) -> Conv2DDeclaration:
    """
    Create a Conv2D layer from parsed parameters with validation.

    Args:
        name: Layer name
        params: Dictionary of parameters from parser
        line: Source line number
        column: Source column number

    Returns:
        Conv2DDeclaration with validated parameters

    Raises:
        ValueError: If parameters are invalid
    """
    try:
        # Extract and validate required parameters
        filters = ConstrainedInt(params["filters"], min_val=1, max_val=2048)

        kernel_size_val = params["kernel_size"]
        if isinstance(kernel_size_val, (list, tuple)):
            kernel_size = KernelSize(tuple(kernel_size_val))
        else:
            kernel_size = KernelSize(kernel_size_val)

        # Extract optional parameters with defaults
        strides_val = params.get("strides", 1)
        if isinstance(strides_val, (list, tuple)):
            strides = StrideValue(tuple(strides_val))
        else:
            strides = StrideValue(strides_val)

        padding = PaddingType(params.get("padding", "valid"))
        activation = ActivationType(params.get("activation", "linear"))
        use_bias = params.get("use_bias", True)

        return Conv2DDeclaration(
            name=name,
            layer_type=LayerType.CONV2D,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            source_line=line,
            source_column=column,
        )

    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid Conv2D parameters for layer '{name}': {str(e)}")


def create_dense_from_params(
    name: str,
    params: Dict[str, Any],
    line: Optional[int] = None,
    column: Optional[int] = None,
) -> DenseDeclaration:
    """Create a Dense layer from parsed parameters with validation."""
    try:
        units = ConstrainedInt(params["units"], min_val=1, max_val=10000)
        activation = ActivationType(params.get("activation", "linear"))
        use_bias = params.get("use_bias", True)

        return DenseDeclaration(
            name=name,
            layer_type=LayerType.DENSE,
            units=units,
            activation=activation,
            use_bias=use_bias,
            source_line=line,
            source_column=column,
        )

    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid Dense parameters for layer '{name}': {str(e)}")


def create_dropout_from_params(
    name: str,
    params: Dict[str, Any],
    line: Optional[int] = None,
    column: Optional[int] = None,
) -> DropoutDeclaration:
    """Create a Dropout layer from parsed parameters with validation."""
    try:
        rate = DropoutRate(params["rate"])

        return DropoutDeclaration(
            name=name,
            layer_type=LayerType.DROPOUT,
            rate=rate,
            source_line=line,
            source_column=column,
        )

    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid Dropout parameters for layer '{name}': {str(e)}")


# ============================================================================
# Integration with Existing Semantic Analyzer
# ============================================================================


def validate_grammar_and_semantics(
    ast: Program, device_config: Optional[ConstraintConfig] = None
) -> ErrorCollector:
    """
    Perform both grammar-level and semantic-level validation.

    Args:
        ast: Parsed AST to validate
        device_config: Device configuration for constraints

    Returns:
        ErrorCollector with all validation results
    """
    # Step 1: Grammar-level validation
    grammar_validator = GrammarSemanticValidator(device_config)
    grammar_errors = grammar_validator.validate_ast(ast)

    # If there are fatal grammar errors, don't proceed to semantic analysis
    if grammar_errors.has_fatal_errors():
        return grammar_errors

    # Step 2: Convert AST to IR and run semantic analysis
    # This would integrate with your existing IR conversion and semantic analyzer
    # For now, return grammar validation results

    return grammar_errors
