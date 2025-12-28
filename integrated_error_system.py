"""EdgeFlow Integrated Error Reporting and Suggestion System

Provides comprehensive error reporting with actionable suggestions, automated fixes,
and integration with the entire EdgeFlow pipeline.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from traceability_system import ProvenanceTracker, ValidationSeverity

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors in the EdgeFlow pipeline."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    DEPLOYMENT = "deployment"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"
    RESOURCE = "resource"


@dataclass
class ErrorSuggestion:
    """A suggestion for fixing an error."""
    description: str
    fix_command: Optional[str] = None
    code_snippet: Optional[str] = None
    documentation_url: Optional[str] = None
    confidence: float = 0.8
    automated: bool = False


@dataclass
class EdgeFlowError:
    """Comprehensive error information with suggestions."""
    error_id: str
    category: ErrorCategory
    severity: ValidationSeverity
    title: str
    message: str
    
    # Location information
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    component: Optional[str] = None
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    related_errors: List[str] = field(default_factory=list)
    
    # Suggestions
    suggestions: List[ErrorSuggestion] = field(default_factory=list)
    
    # Metadata
    first_seen: Optional[str] = None
    frequency: int = 1
    resolved: bool = False


class ErrorKnowledgeBase:
    """Knowledge base of common errors and their solutions."""
    
    def __init__(self):
        self.error_patterns = {}
        self.solution_templates = {}
        self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> None:
        """Load error patterns and solutions."""
        self.error_patterns = {
            # Syntax errors
            r"Invalid assignment syntax": {
                "category": ErrorCategory.SYNTAX,
                "suggestions": [
                    ErrorSuggestion(
                        description="Use 'key = value' format for assignments",
                        code_snippet="model_path = \"path/to/model.tflite\"",
                        confidence=0.9,
                        automated=True
                    )
                ]
            },
            
            # Parameter validation errors
            r"Invalid quantization type: (.+)": {
                "category": ErrorCategory.VALIDATION,
                "suggestions": [
                    ErrorSuggestion(
                        description="Use a valid quantization type",
                        code_snippet="quantize = int8  # or float16, dynamic, none",
                        documentation_url="https://edgeflow.docs/quantization",
                        confidence=0.95
                    )
                ]
            },
            
            # Device compatibility errors
            r"Model size (.+)MB exceeds device limit (.+)MB": {
                "category": ErrorCategory.COMPATIBILITY,
                "suggestions": [
                    ErrorSuggestion(
                        description="Enable aggressive quantization to reduce model size",
                        fix_command="echo 'quantize = int8' >> config.ef",
                        confidence=0.8,
                        automated=True
                    ),
                    ErrorSuggestion(
                        description="Enable model pruning to reduce parameters",
                        fix_command="echo 'enable_pruning = true' >> config.ef",
                        confidence=0.7,
                        automated=True
                    )
                ]
            },
            
            # Memory errors
            r"Insufficient memory for model": {
                "category": ErrorCategory.RESOURCE,
                "suggestions": [
                    ErrorSuggestion(
                        description="Reduce model size with quantization",
                        code_snippet="quantize = int8\nenable_pruning = true",
                        confidence=0.9
                    ),
                    ErrorSuggestion(
                        description="Use a smaller model architecture",
                        documentation_url="https://edgeflow.docs/model-selection",
                        confidence=0.6
                    )
                ]
            },
            
            # Performance warnings
            r"Latency (.+)ms exceeds target (.+)ms": {
                "category": ErrorCategory.PERFORMANCE,
                "suggestions": [
                    ErrorSuggestion(
                        description="Enable operator fusion for better performance",
                        fix_command="echo 'enable_fusion = true' >> config.ef",
                        confidence=0.8,
                        automated=True
                    ),
                    ErrorSuggestion(
                        description="Use GPU acceleration if available",
                        code_snippet="target_device = gpu",
                        confidence=0.7
                    )
                ]
            }
        }
    
    def find_suggestions(self, error_message: str) -> List[ErrorSuggestion]:
        """Find suggestions for an error message."""
        suggestions = []
        
        for pattern, info in self.error_patterns.items():
            if re.search(pattern, error_message, re.IGNORECASE):
                suggestions.extend(info["suggestions"])
        
        return suggestions


class IntegratedErrorReporter:
    """Integrated error reporting system with suggestions."""
    
    def __init__(self, tracker: Optional[ProvenanceTracker] = None):
        self.tracker = tracker or ProvenanceTracker()
        self.knowledge_base = ErrorKnowledgeBase()
        self.error_history: Dict[str, EdgeFlowError] = {}
        self.session_errors: List[EdgeFlowError] = []
    
    def report_error(
        self,
        category: ErrorCategory,
        severity: ValidationSeverity,
        title: str,
        message: str,
        source_file: Optional[str] = None,
        line_number: Optional[int] = None,
        component: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EdgeFlowError:
        """Report a new error with automatic suggestion generation."""
        
        # Generate error ID
        error_id = f"{category.value}_{hash(message) % 10000:04d}"
        
        # Check if this is a recurring error
        existing_error = self.error_history.get(error_id)
        if existing_error:
            existing_error.frequency += 1
            return existing_error
        
        # Create new error
        error = EdgeFlowError(
            error_id=error_id,
            category=category,
            severity=severity,
            title=title,
            message=message,
            source_file=source_file,
            line_number=line_number,
            component=component,
            context=context or {},
        )
        
        # Generate suggestions
        error.suggestions = self.knowledge_base.find_suggestions(message)
        
        # Add context-specific suggestions
        self._add_context_suggestions(error)
        
        # Store error
        self.error_history[error_id] = error
        self.session_errors.append(error)
        
        # Log error
        self._log_error(error)
        
        return error
    
    def _add_context_suggestions(self, error: EdgeFlowError) -> None:
        """Add context-specific suggestions based on error details."""
        
        # File-specific suggestions
        if error.source_file and error.source_file.endswith('.ef'):
            error.suggestions.append(ErrorSuggestion(
                description=f"Validate the entire file: edgeflow validate {error.source_file}",
                fix_command=f"python interactive_validator.py validate-file {error.source_file}",
                confidence=0.6,
                automated=True
            ))
        
        # Component-specific suggestions
        if error.component == "optimizer":
            error.suggestions.append(ErrorSuggestion(
                description="Try a different optimization strategy",
                code_snippet="optimization_level = basic  # or aggressive, balanced",
                confidence=0.5
            ))
        
        # Severity-specific suggestions
        if error.severity == ValidationSeverity.CRITICAL:
            error.suggestions.insert(0, ErrorSuggestion(
                description="This is a critical error that must be fixed before proceeding",
                confidence=1.0
            ))
    
    def _log_error(self, error: EdgeFlowError) -> None:
        """Log error to appropriate logging level."""
        location = ""
        if error.source_file:
            location = f" in {error.source_file}"
            if error.line_number:
                location += f":{error.line_number}"
        
        log_message = f"[{error.category.value.upper()}] {error.title}{location}: {error.message}"
        
        if error.severity == ValidationSeverity.CRITICAL:
            logger.critical(log_message)
        elif error.severity == ValidationSeverity.ERROR:
            logger.error(log_message)
        elif error.severity == ValidationSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of errors in current session."""
        error_counts = {}
        for category in ErrorCategory:
            error_counts[category.value] = sum(
                1 for e in self.session_errors if e.category == category
            )
        
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = sum(
                1 for e in self.session_errors if e.severity == severity
            )
        
        return {
            "total_errors": len(self.session_errors),
            "by_category": error_counts,
            "by_severity": severity_counts,
            "critical_errors": [e for e in self.session_errors if e.severity == ValidationSeverity.CRITICAL],
            "automated_fixes_available": sum(
                1 for e in self.session_errors 
                for s in e.suggestions if s.automated
            )
        }
    
    def apply_automated_fixes(self, error_ids: Optional[List[str]] = None) -> Dict[str, bool]:
        """Apply automated fixes for specified errors."""
        results = {}
        
        errors_to_fix = []
        if error_ids:
            errors_to_fix = [self.error_history[eid] for eid in error_ids if eid in self.error_history]
        else:
            errors_to_fix = [e for e in self.session_errors if any(s.automated for s in e.suggestions)]
        
        for error in errors_to_fix:
            for suggestion in error.suggestions:
                if suggestion.automated and suggestion.fix_command:
                    try:
                        # Execute fix command
                        import subprocess
                        result = subprocess.run(
                            suggestion.fix_command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        success = result.returncode == 0
                        results[error.error_id] = success
                        
                        if success:
                            error.resolved = True
                            logger.info(f"Applied automated fix for {error.error_id}")
                        else:
                            logger.warning(f"Automated fix failed for {error.error_id}: {result.stderr}")
                    
                    except Exception as e:
                        results[error.error_id] = False
                        logger.error(f"Failed to apply automated fix for {error.error_id}: {e}")
        
        return results
    
    def generate_error_report(self, output_file: str = "error_report.md") -> str:
        """Generate comprehensive error report."""
        
        summary = self.get_session_summary()
        
        report_lines = [
            "# EdgeFlow Error Report",
            "",
            f"**Total Errors:** {summary['total_errors']}",
            f"**Critical Errors:** {len(summary['critical_errors'])}",
            f"**Automated Fixes Available:** {summary['automated_fixes_available']}",
            "",
            "## Error Summary by Category",
            ""
        ]
        
        for category, count in summary['by_category'].items():
            if count > 0:
                report_lines.append(f"- **{category.title()}:** {count}")
        
        report_lines.extend([
            "",
            "## Error Summary by Severity",
            ""
        ])
        
        for severity, count in summary['by_severity'].items():
            if count > 0:
                report_lines.append(f"- **{severity.title()}:** {count}")
        
        # Detailed error information
        if summary['critical_errors']:
            report_lines.extend([
                "",
                "## Critical Errors (Must Fix)",
                ""
            ])
            
            for error in summary['critical_errors']:
                report_lines.extend(self._format_error_for_report(error))
        
        # All other errors
        other_errors = [e for e in self.session_errors if e.severity != ValidationSeverity.CRITICAL]
        if other_errors:
            report_lines.extend([
                "",
                "## Other Errors and Warnings",
                ""
            ])
            
            for error in other_errors:
                report_lines.extend(self._format_error_for_report(error))
        
        # Write report
        report_content = "\n".join(report_lines)
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated error report: {output_file}")
        return report_content
    
    def _format_error_for_report(self, error: EdgeFlowError) -> List[str]:
        """Format an error for inclusion in the report."""
        lines = [
            f"### {error.title}",
            "",
            f"**Category:** {error.category.value}  ",
            f"**Severity:** {error.severity.value}  ",
        ]
        
        if error.source_file:
            location = error.source_file
            if error.line_number:
                location += f":{error.line_number}"
            lines.append(f"**Location:** {location}  ")
        
        if error.component:
            lines.append(f"**Component:** {error.component}  ")
        
        lines.extend([
            "",
            f"**Message:** {error.message}",
            ""
        ])
        
        if error.suggestions:
            lines.extend([
                "**Suggestions:**",
                ""
            ])
            
            for i, suggestion in enumerate(error.suggestions, 1):
                lines.append(f"{i}. {suggestion.description}")
                
                if suggestion.code_snippet:
                    lines.extend([
                        "   ```",
                        f"   {suggestion.code_snippet}",
                        "   ```"
                    ])
                
                if suggestion.fix_command and suggestion.automated:
                    lines.append(f"   **Auto-fix:** `{suggestion.fix_command}`")
                
                if suggestion.documentation_url:
                    lines.append(f"   **Documentation:** {suggestion.documentation_url}")
                
                lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines


# Global error reporter instance
_global_reporter: Optional[IntegratedErrorReporter] = None


def get_error_reporter() -> IntegratedErrorReporter:
    """Get or create the global error reporter."""
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = IntegratedErrorReporter()
    return _global_reporter


def report_error(
    category: ErrorCategory,
    severity: ValidationSeverity,
    title: str,
    message: str,
    **kwargs
) -> EdgeFlowError:
    """Report an error using the global reporter."""
    reporter = get_error_reporter()
    return reporter.report_error(category, severity, title, message, **kwargs)


if __name__ == "__main__":
    # Example usage
    reporter = IntegratedErrorReporter()
    
    # Report some example errors
    reporter.report_error(
        ErrorCategory.VALIDATION,
        ValidationSeverity.ERROR,
        "Invalid Quantization Type",
        "Invalid quantization type: float32",
        source_file="model.ef",
        line_number=5
    )
    
    reporter.report_error(
        ErrorCategory.COMPATIBILITY,
        ValidationSeverity.WARNING,
        "Model Size Exceeds Limit",
        "Model size 10.5MB exceeds device limit 8.0MB",
        component="device_validator"
    )
    
    # Generate report
    report = reporter.generate_error_report()
    print("Generated error report with suggestions!")
    
    # Show session summary
    summary = reporter.get_session_summary()
    print(f"Session summary: {summary['total_errors']} total errors")
