"""
Error types and severity levels for semantic analysis.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorSeverity(Enum):
    """Severity levels for semantic analysis errors."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class ErrorType(Enum):
    """Types of semantic analysis errors."""

    SHAPE_MISMATCH = "ShapeMismatch"
    DATATYPE_MISMATCH = "DatatypeMismatch"
    PARAM_RANGE = "ParamRange"
    FORBIDDEN_CONFIG = "ForbiddenConfig"
    RESOURCE_LIMIT = "ResourceLimit"
    GRAPH_CYCLE = "GraphCycle"
    CONNECTIVITY = "Connectivity"
    MISSING_LAYER = "MissingLayer"
    INVALID_SEQUENCE = "InvalidSequence"
    UNSUPPORTED_OP = "UnsupportedOp"
    DEVICE_INCOMPATIBLE = "DeviceIncompatible"


@dataclass
class SourceLocation:
    """Represents a location in the DSL source code."""

    line: int
    column: int
    file_path: Optional[str] = None

    def __str__(self) -> str:
        location = f"line {self.line}, column {self.column}"
        if self.file_path:
            location = f"{self.file_path}:{location}"
        return location


@dataclass
class SemanticError:
    """Represents a semantic analysis error or warning."""

    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    node_id: Optional[str] = None
    layer_name: Optional[str] = None
    location: Optional[SourceLocation] = None
    context: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        """Format error for display."""
        prefix = f"[{self.severity.value.upper()}]"

        if self.location:
            prefix += f" at {self.location}"
        elif self.layer_name:
            prefix += f" in layer '{self.layer_name}'"
        elif self.node_id:
            prefix += f" in node '{self.node_id}'"

        result = f"{prefix}: {self.message}"

        if self.suggestion:
            result += f"\n  Suggestion: {self.suggestion}"

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "type": self.error_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "node_id": self.node_id,
            "layer_name": self.layer_name,
            "location": {
                "line": self.location.line,
                "column": self.location.column,
                "file_path": self.location.file_path,
            }
            if self.location
            else None,
            "context": self.context,
            "suggestion": self.suggestion,
        }


class ErrorCollector:
    """Collects and manages semantic analysis errors."""

    def __init__(self):
        self.errors: List[SemanticError] = []
        self.error_counts = {severity: 0 for severity in ErrorSeverity}

    def add_error(self, error: SemanticError) -> None:
        """Add an error to the collection."""
        self.errors.append(error)
        self.error_counts[error.severity] += 1

    def add_shape_mismatch(
        self,
        expected_shape: tuple,
        actual_shape: tuple,
        node_id: str = None,
        layer_name: str = None,
        location: SourceLocation = None,
    ) -> None:
        """Add a shape mismatch error."""
        message = f"Expected input shape {expected_shape}, got {actual_shape}"
        suggestion = f"Ensure the previous layer outputs shape {expected_shape}"

        error = SemanticError(
            error_type=ErrorType.SHAPE_MISMATCH,
            severity=ErrorSeverity.ERROR,
            message=message,
            node_id=node_id,
            layer_name=layer_name,
            location=location,
            context={"expected": expected_shape, "actual": actual_shape},
            suggestion=suggestion,
        )
        self.add_error(error)

    def add_param_range_error(
        self,
        param_name: str,
        value: Any,
        min_val: Any,
        max_val: Any,
        node_id: str = None,
        layer_name: str = None,
        location: SourceLocation = None,
    ) -> None:
        """Add a parameter range error."""
        message = f"Parameter '{param_name}' value {value} is out of range [{min_val}, {max_val}]"
        suggestion = f"Set '{param_name}' to a value between {min_val} and {max_val}"

        error = SemanticError(
            error_type=ErrorType.PARAM_RANGE,
            severity=ErrorSeverity.ERROR,
            message=message,
            node_id=node_id,
            layer_name=layer_name,
            location=location,
            context={
                "param": param_name,
                "value": value,
                "min": min_val,
                "max": max_val,
            },
            suggestion=suggestion,
        )
        self.add_error(error)

    def add_forbidden_config(
        self,
        config_description: str,
        node_id: str = None,
        layer_name: str = None,
        location: SourceLocation = None,
        suggestion: str = None,
    ) -> None:
        """Add a forbidden configuration error."""
        error = SemanticError(
            error_type=ErrorType.FORBIDDEN_CONFIG,
            severity=ErrorSeverity.ERROR,
            message=f"Forbidden configuration: {config_description}",
            node_id=node_id,
            layer_name=layer_name,
            location=location,
            suggestion=suggestion,
        )
        self.add_error(error)

    def add_resource_limit_error(
        self,
        resource_type: str,
        usage: float,
        limit: float,
        node_id: str = None,
        layer_name: str = None,
        location: SourceLocation = None,
    ) -> None:
        """Add a resource limit error."""
        message = f"{resource_type} usage {usage:.2f} exceeds limit {limit:.2f}"
        suggestion = f"Reduce {resource_type} usage or increase device limits"

        error = SemanticError(
            error_type=ErrorType.RESOURCE_LIMIT,
            severity=ErrorSeverity.ERROR,
            message=message,
            node_id=node_id,
            layer_name=layer_name,
            location=location,
            context={"resource_type": resource_type, "usage": usage, "limit": limit},
            suggestion=suggestion,
        )
        self.add_error(error)

    def has_errors(self) -> bool:
        """Check if there are any errors (not warnings or info)."""
        return (
            self.error_counts[ErrorSeverity.ERROR] > 0
            or self.error_counts[ErrorSeverity.FATAL] > 0
        )

    def has_fatal_errors(self) -> bool:
        """Check if there are any fatal errors."""
        return self.error_counts[ErrorSeverity.FATAL] > 0

    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[SemanticError]:
        """Get all errors of a specific severity."""
        return [error for error in self.errors if error.severity == severity]

    def print_summary(self) -> None:
        """Print a summary of all errors."""
        if not self.errors:
            print("✅ No semantic analysis errors found.")
            return

        print(f"\n📊 Semantic Analysis Summary:")
        print(f"   Errors: {self.error_counts[ErrorSeverity.ERROR]}")
        print(f"   Warnings: {self.error_counts[ErrorSeverity.WARNING]}")
        print(f"   Info: {self.error_counts[ErrorSeverity.INFO]}")
        print(f"   Fatal: {self.error_counts[ErrorSeverity.FATAL]}")

        print(f"\n📝 Detailed Report:")
        for error in self.errors:
            print(f"  {error}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert error collection to dictionary."""
        return {
            "summary": dict(self.error_counts),
            "errors": [error.to_dict() for error in self.errors],
        }
