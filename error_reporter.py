"""EdgeFlow Error Reporting System

This module provides comprehensive error reporting with clear, actionable messages,
suggestions for fixes, and context-aware help for EdgeFlow DSL configurations.

Key Features:
- Human-readable error messages
- Actionable suggestions and fixes
- Context-aware help
- Error categorization and severity
- Integration with validation results
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from static_validator import ValidationCategory, ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)


class ErrorContext(Enum):
    """Context where the error occurred."""

    PARSING = "parsing"
    VALIDATION = "validation"
    COMPILATION = "compilation"
    OPTIMIZATION = "optimization"
    CODE_GENERATION = "code_generation"
    RUNTIME = "runtime"


@dataclass
class ErrorReport:
    """Comprehensive error report with context and suggestions."""

    error_id: str
    severity: ValidationSeverity
    category: ValidationCategory
    context: ErrorContext
    title: str
    message: str
    explanation: str
    suggestions: List[str]
    code_examples: List[str]
    related_docs: List[str]
    parameter: Optional[str] = None
    current_value: Optional[Any] = None
    suggested_value: Optional[Any] = None
    line_number: Optional[int] = None
    impact: Optional[str] = None


class EdgeFlowErrorReporter:
    """Generates comprehensive error reports for EdgeFlow configurations."""

    def __init__(self):
        self.error_templates = self._initialize_error_templates()
        self.suggestion_templates = self._initialize_suggestion_templates()
        self.code_examples = self._initialize_code_examples()

    def _initialize_error_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error message templates."""
        return {
            "missing_model": {
                "title": "Missing Model File",
                "message": "No model file specified in configuration",
                "explanation": "EdgeFlow requires a model file to optimize. This should be the path to your ML model.",
                "suggestions": [
                    "Add a 'model' parameter to your configuration",
                    "Specify the full path to your model file",
                    "Ensure the model file exists and is accessible",
                ],
                "code_examples": [
                    'model = "path/to/your/model.tflite"',
                    'model = "models/mobilenet_v2.h5"',
                ],
                "related_docs": ["model_formats.md", "getting_started.md"],
            },
            "invalid_device": {
                "title": "Unsupported Target Device",
                "message": "The specified target device is not supported",
                "explanation": "EdgeFlow supports specific edge devices with known capabilities and constraints.",
                "suggestions": [
                    "Use a supported device from the list",
                    "Check device capabilities before setting parameters",
                    "Consider using 'cpu' as a fallback option",
                ],
                "code_examples": [
                    'target_device = "raspberry_pi"',
                    'target_device = "jetson_nano"',
                    'target_device = "cpu"',
                ],
                "related_docs": ["supported_devices.md", "device_capabilities.md"],
            },
            "incompatible_quantization": {
                "title": "Incompatible Quantization Setting",
                "message": "The quantization type is not compatible with the target device or model format",
                "explanation": "Different devices and model formats have different quantization capabilities.",
                "suggestions": [
                    "Check device quantization support",
                    "Verify model format compatibility",
                    "Consider alternative quantization types",
                ],
                "code_examples": [
                    'quantize = "int8"  # For most devices',
                    'quantize = "float16"  # For newer devices',
                    'quantize = "none"  # No quantization',
                ],
                "related_docs": ["quantization_guide.md", "device_capabilities.md"],
            },
            "memory_limit_exceeded": {
                "title": "Memory Limit Too High",
                "message": "The specified memory limit exceeds the device capacity",
                "explanation": "Each device has physical memory limitations that cannot be exceeded.",
                "suggestions": [
                    "Reduce the memory limit to fit device capacity",
                    "Consider using a more powerful device",
                    "Optimize your model to use less memory",
                ],
                "code_examples": [
                    "memory_limit = 256  # For Raspberry Pi",
                    "memory_limit = 1024  # For Jetson Nano",
                    "memory_limit = 2048  # For Jetson Xavier",
                ],
                "related_docs": ["memory_optimization.md", "device_capabilities.md"],
            },
            "buffer_size_too_large": {
                "title": "Buffer Size Too Large",
                "message": "The buffer size may cause memory pressure on the target device",
                "explanation": "Large buffer sizes can consume too much memory and slow down performance.",
                "suggestions": [
                    "Reduce buffer size to recommended value",
                    "Consider device memory capacity",
                    "Balance between throughput and memory usage",
                ],
                "code_examples": [
                    "buffer_size = 16  # For Raspberry Pi",
                    "buffer_size = 32  # For Jetson Nano",
                    "buffer_size = 64  # For high-end devices",
                ],
                "related_docs": ["buffer_optimization.md", "performance_tuning.md"],
            },
            "model_format_incompatible": {
                "title": "Model Format Not Compatible",
                "message": "The model format is not compatible with the specified optimizations",
                "explanation": "Different model formats support different optimization techniques.",
                "suggestions": [
                    "Convert model to a compatible format",
                    "Disable incompatible optimizations",
                    "Use a different model format",
                ],
                "code_examples": [
                    "# Convert PyTorch to TensorFlow Lite",
                    "import torch",
                    "model = torch.load('model.pth')",
                    "# Convert and save as .tflite",
                ],
                "related_docs": ["model_conversion.md", "supported_formats.md"],
            },
            "fusion_not_supported": {
                "title": "Operation Fusion Not Supported",
                "message": "The target device or model format doesn't support operation fusion",
                "explanation": "Operation fusion requires specific hardware or software support.",
                "suggestions": [
                    "Disable fusion for this configuration",
                    "Use a device that supports fusion",
                    "Convert to a compatible model format",
                ],
                "code_examples": [
                    "enable_fusion = false  # Disable fusion",
                    'target_device = "jetson_xavier"  # Use fusion-capable device',
                ],
                "related_docs": ["fusion_optimization.md", "device_capabilities.md"],
            },
            "optimization_conflict": {
                "title": "Optimization Goal Conflict",
                "message": "The optimization settings conflict with each other",
                "explanation": "Some optimization techniques work against each other or have trade-offs.",
                "suggestions": [
                    "Choose a primary optimization goal",
                    "Balance conflicting optimizations",
                    "Consider the target use case",
                ],
                "code_examples": [
                    'optimize_for = "latency"  # Focus on speed',
                    'optimize_for = "memory"  # Focus on memory usage',
                    'optimize_for = "balanced"  # Balance all factors',
                ],
                "related_docs": ["optimization_strategies.md", "performance_tuning.md"],
            },
            "invalid_parameter_value": {
                "title": "Invalid Parameter Value",
                "message": "The parameter value is outside the allowed range or format",
                "explanation": "Parameters have specific value ranges and formats for valid operation.",
                "suggestions": [
                    "Check parameter documentation",
                    "Use values within the allowed range",
                    "Verify parameter format and type",
                ],
                "code_examples": [
                    "buffer_size = 32  # Integer between 1-256",
                    "memory_limit = 512  # Positive number in MB",
                    "pruning_sparsity = 0.5  # Float between 0.0-1.0",
                ],
                "related_docs": ["parameter_reference.md", "configuration_guide.md"],
            },
        }

    def _initialize_suggestion_templates(self) -> Dict[str, List[str]]:
        """Initialize suggestion templates for common fixes."""
        return {
            "device_suggestions": {
                "raspberry_pi": [
                    "Use INT8 quantization for best performance",
                    "Keep buffer size under 32",
                    "Set memory limit to 256MB or less",
                    "Enable fusion for TensorFlow Lite models",
                ],
                "jetson_nano": [
                    "Use FP16 quantization for better accuracy",
                    "Buffer size can be up to 64",
                    "Memory limit can be up to 1GB",
                    "Enable both fusion and pruning",
                ],
                "jetson_xavier": [
                    "Use FP16 quantization for optimal performance",
                    "Buffer size can be up to 128",
                    "Memory limit can be up to 2GB",
                    "Enable all optimizations for best results",
                ],
                "cortex_m4": [
                    "Use INT8 quantization only",
                    "Keep buffer size under 8",
                    "Set memory limit to 128MB or less",
                    "Disable fusion (not supported)",
                ],
                "cpu": [
                    "Use FP16 quantization for accuracy",
                    "Buffer size can be up to 128",
                    "Memory limit can be up to 4GB",
                    "Enable fusion for better performance",
                ],
                "gpu": [
                    "Use FP16 quantization for optimal performance",
                    "Buffer size can be up to 256",
                    "Memory limit can be up to 8GB",
                    "Enable all optimizations",
                ],
            },
            "model_format_suggestions": {
                ".tflite": [
                    "Best for edge deployment",
                    "Supports INT8 and FP16 quantization",
                    "Works well with operation fusion",
                    "Recommended for Raspberry Pi and Jetson",
                ],
                ".h5": [
                    "Good for development and testing",
                    "Supports INT8 quantization",
                    "Works well with pruning",
                    "Best for CPU and GPU deployment",
                ],
                ".onnx": [
                    "Cross-platform compatibility",
                    "Supports both INT8 and FP16 quantization",
                    "Works well with all optimizations",
                    "Recommended for Jetson Xavier and GPU",
                ],
                ".pb": [
                    "TensorFlow SavedModel format",
                    "Limited quantization support",
                    "Good for CPU and GPU deployment",
                    "Not recommended for edge devices",
                ],
                ".pth": [
                    "PyTorch model format",
                    "No quantization support",
                    "No fusion support",
                    "Convert to TensorFlow Lite for edge deployment",
                ],
            },
        }

    def _initialize_code_examples(self) -> Dict[str, List[str]]:
        """Initialize code examples for common configurations."""
        return {
            "raspberry_pi_config": [
                "# Optimized configuration for Raspberry Pi",
                'model = "mobilenet_v2.tflite"',
                "quantize = int8",
                "target_device = raspberry_pi",
                "enable_fusion = true",
                "buffer_size = 16",
                "memory_limit = 256",
                "optimize_for = latency",
            ],
            "jetson_nano_config": [
                "# Optimized configuration for Jetson Nano",
                'model = "efficientnet_b0.tflite"',
                "quantize = float16",
                "target_device = jetson_nano",
                "enable_fusion = true",
                "enable_pruning = true",
                "pruning_sparsity = 0.3",
                "buffer_size = 32",
                "memory_limit = 1024",
                "optimize_for = balanced",
            ],
            "jetson_xavier_config": [
                "# Optimized configuration for Jetson Xavier",
                'model = "resnet50.onnx"',
                "quantize = float16",
                "target_device = jetson_xavier",
                "enable_fusion = true",
                "enable_pruning = true",
                "pruning_sparsity = 0.5",
                "buffer_size = 64",
                "memory_limit = 2048",
                "optimize_for = accuracy",
            ],
            "cortex_m4_config": [
                "# Optimized configuration for Cortex-M4",
                'model = "tiny_model.tflite"',
                "quantize = int8",
                "target_device = cortex_m4",
                "enable_fusion = false",
                "buffer_size = 4",
                "memory_limit = 128",
                "optimize_for = memory",
            ],
        }

    def generate_error_report(
        self, issue: ValidationIssue, context: ErrorContext = ErrorContext.VALIDATION
    ) -> ErrorReport:
        """Generate a comprehensive error report from a validation issue.

        Args:
            issue: Validation issue to report
            context: Context where the error occurred

        Returns:
            Comprehensive error report
        """
        # Determine error template based on issue characteristics
        template_key = self._identify_error_template(issue)
        template = self.error_templates.get(
            template_key, self.error_templates["invalid_parameter_value"]
        )

        # Generate suggestions based on context
        suggestions = self._generate_contextual_suggestions(issue, template)

        # Generate code examples
        code_examples = self._generate_code_examples(issue, template)

        # Generate related documentation links
        related_docs = self._generate_related_docs(issue, template)

        return ErrorReport(
            error_id=f"{issue.category.value}_{template_key}",
            severity=issue.severity,
            category=issue.category,
            context=context,
            title=template["title"],
            message=issue.message,
            explanation=template["explanation"],
            suggestions=suggestions,
            code_examples=code_examples,
            related_docs=related_docs,
            parameter=issue.parameter,
            current_value=issue.current_value,
            suggested_value=issue.suggested_value,
            line_number=issue.line_number,
            impact=issue.impact,
        )

    def _identify_error_template(self, issue: ValidationIssue) -> str:
        """Identify the appropriate error template for an issue."""
        if issue.parameter == "model" and issue.severity == ValidationSeverity.ERROR:
            return "missing_model"
        elif issue.parameter == "target_device":
            return "invalid_device"
        elif issue.parameter == "quantize":
            return "incompatible_quantization"
        elif issue.parameter == "memory_limit":
            return "memory_limit_exceeded"
        elif issue.parameter == "buffer_size":
            return "buffer_size_too_large"
        elif (
            issue.category == ValidationCategory.COMPATIBILITY
            and "format" in issue.message.lower()
        ):
            return "model_format_incompatible"
        elif issue.parameter == "enable_fusion":
            return "fusion_not_supported"
        elif issue.category == ValidationCategory.PERFORMANCE:
            return "optimization_conflict"
        else:
            return "invalid_parameter_value"

    def _generate_contextual_suggestions(
        self, issue: ValidationIssue, template: Dict[str, Any]
    ) -> List[str]:
        """Generate contextual suggestions for fixing the issue."""
        suggestions = template["suggestions"].copy()

        # Add device-specific suggestions
        if issue.parameter == "target_device" and issue.current_value:
            device_suggestions = self.suggestion_templates["device_suggestions"].get(
                str(issue.current_value), []
            )
            suggestions.extend(device_suggestions)

        # Add model format suggestions
        if issue.parameter == "model" and issue.current_value:
            model_path = str(issue.current_value)
            if "." in model_path:
                model_ext = "." + model_path.split(".")[-1].lower()
                format_suggestions = self.suggestion_templates[
                    "model_format_suggestions"
                ].get(model_ext, [])
                suggestions.extend(format_suggestions)

        # Add specific suggestions based on suggested value
        if issue.suggested_value is not None:
            suggestions.append(
                f"Try setting {issue.parameter} to: {issue.suggested_value}"
            )

        return suggestions

    def _generate_code_examples(
        self, issue: ValidationIssue, template: Dict[str, Any]
    ) -> List[str]:
        """Generate relevant code examples for the issue."""
        examples = template["code_examples"].copy()

        # Add device-specific examples
        if issue.parameter == "target_device" and issue.suggested_value:
            device = str(issue.suggested_value)
            device_examples = self.code_examples.get(f"{device}_config", [])
            examples.extend(device_examples)

        # Add parameter-specific examples
        if issue.parameter and issue.suggested_value is not None:
            examples.append(f"{issue.parameter} = {issue.suggested_value}")

        return examples

    def _generate_related_docs(
        self, issue: ValidationIssue, template: Dict[str, Any]
    ) -> List[str]:
        """Generate related documentation links."""
        docs = template["related_docs"].copy()

        # Add category-specific docs
        if issue.category == ValidationCategory.COMPATIBILITY:
            docs.append("compatibility_matrix.md")
        elif issue.category == ValidationCategory.PERFORMANCE:
            docs.append("performance_optimization.md")
        elif issue.category == ValidationCategory.SYNTAX:
            docs.append("configuration_syntax.md")

        return docs

    def format_error_report(
        self, report: ErrorReport, include_details: bool = True
    ) -> str:
        """Format an error report for display.

        Args:
            report: Error report to format
            include_details: Whether to include detailed information

        Returns:
            Formatted error message
        """
        lines = []

        # Header
        severity_icon = "❌" if report.severity == ValidationSeverity.ERROR else "⚠️"
        lines.append(f"{severity_icon} {report.title}")
        lines.append("=" * (len(report.title) + 3))

        # Main message
        lines.append(f"\n{report.message}")

        # Explanation
        lines.append(f"\n📖 Explanation:")
        lines.append(f"   {report.explanation}")

        # Impact
        if report.impact:
            lines.append(f"\n⚡ Impact:")
            lines.append(f"   {report.impact}")

        # Suggestions
        if report.suggestions:
            lines.append(f"\n💡 Suggestions:")
            for i, suggestion in enumerate(report.suggestions, 1):
                lines.append(f"   {i}. {suggestion}")

        # Code examples
        if report.code_examples and include_details:
            lines.append(f"\n💻 Code Examples:")
            for example in report.code_examples:
                lines.append(f"   {example}")

        # Parameter information
        if report.parameter and include_details:
            lines.append(f"\n🔧 Parameter: {report.parameter}")
            if report.current_value is not None:
                lines.append(f"   Current value: {report.current_value}")
            if report.suggested_value is not None:
                lines.append(f"   Suggested value: {report.suggested_value}")

        # Related documentation
        if report.related_docs and include_details:
            lines.append(f"\n📚 Related Documentation:")
            for doc in report.related_docs:
                lines.append(f"   - {doc}")

        return "\n".join(lines)

    def generate_summary_report(
        self,
        issues: List[ValidationIssue],
        context: ErrorContext = ErrorContext.VALIDATION,
    ) -> str:
        """Generate a summary report for multiple issues.

        Args:
            issues: List of validation issues
            context: Context where the errors occurred

        Returns:
            Formatted summary report
        """
        if not issues:
            return "✅ No validation issues found!"

        lines = []

        # Summary header
        error_count = sum(
            1 for issue in issues if issue.severity == ValidationSeverity.ERROR
        )
        warning_count = sum(
            1 for issue in issues if issue.severity == ValidationSeverity.WARNING
        )

        lines.append("📋 Validation Summary")
        lines.append("=" * 20)
        lines.append(f"Errors: {error_count}")
        lines.append(f"Warnings: {warning_count}")
        lines.append(f"Total Issues: {len(issues)}")

        # Group issues by severity
        errors = [
            issue for issue in issues if issue.severity == ValidationSeverity.ERROR
        ]
        warnings = [
            issue for issue in issues if issue.severity == ValidationSeverity.WARNING
        ]

        # Report errors
        if errors:
            lines.append(f"\n❌ Errors ({len(errors)}):")
            for i, issue in enumerate(errors, 1):
                lines.append(f"   {i}. {issue.message}")
                if issue.parameter:
                    lines.append(f"      Parameter: {issue.parameter}")

        # Report warnings
        if warnings:
            lines.append(f"\n⚠️ Warnings ({len(warnings)}):")
            for i, issue in enumerate(warnings, 1):
                lines.append(f"   {i}. {issue.message}")
                if issue.parameter:
                    lines.append(f"      Parameter: {issue.parameter}")

        # Overall recommendation
        if error_count > 0:
            lines.append(f"\n🚨 Action Required:")
            lines.append("   Fix all errors before proceeding with compilation.")
        elif warning_count > 0:
            lines.append(f"\n💡 Recommendation:")
            lines.append("   Review warnings and consider optimizations.")
        else:
            lines.append(f"\n✅ Configuration looks good!")

        return "\n".join(lines)


def generate_error_report(
    issue: ValidationIssue, context: ErrorContext = ErrorContext.VALIDATION
) -> ErrorReport:
    """Generate a comprehensive error report.

    Args:
        issue: Validation issue to report
        context: Context where the error occurred

    Returns:
        Comprehensive error report
    """
    reporter = EdgeFlowErrorReporter()
    return reporter.generate_error_report(issue, context)


def format_error_report(report: ErrorReport, include_details: bool = True) -> str:
    """Format an error report for display.

    Args:
        report: Error report to format
        include_details: Whether to include detailed information

    Returns:
        Formatted error message
    """
    reporter = EdgeFlowErrorReporter()
    return reporter.format_error_report(report, include_details)


def generate_summary_report(
    issues: List[ValidationIssue], context: ErrorContext = ErrorContext.VALIDATION
) -> str:
    """Generate a summary report for multiple issues.

    Args:
        issues: List of validation issues
        context: Context where the errors occurred

    Returns:
        Formatted summary report
    """
    reporter = EdgeFlowErrorReporter()
    return reporter.generate_summary_report(issues, context)


if __name__ == "__main__":
    # Test the error reporter
    from static_validator import ValidationCategory, ValidationIssue, ValidationSeverity

    test_issues = [
        ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category=ValidationCategory.SYNTAX,
            message="Missing required parameter: 'model'",
            parameter="model",
            suggested_value="path/to/model.tflite",
        ),
        ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category=ValidationCategory.COMPATIBILITY,
            message="Device cortex_m4 doesn't support FP16 quantization",
            parameter="quantize",
            current_value="float16",
            suggested_value="int8",
        ),
        ValidationIssue(
            severity=ValidationSeverity.WARNING,
            category=ValidationCategory.PERFORMANCE,
            message="Buffer size 64 may be too large for raspberry_pi",
            parameter="buffer_size",
            current_value=64,
            suggested_value=16,
        ),
    ]

    reporter = EdgeFlowErrorReporter()

    print("=== Individual Error Reports ===")
    for issue in test_issues:
        report = reporter.generate_error_report(issue)
        print(reporter.format_error_report(report))
        print("\n" + "=" * 50 + "\n")

    print("=== Summary Report ===")
    summary = reporter.generate_summary_report(test_issues)
    print(summary)
