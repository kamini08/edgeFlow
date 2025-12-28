"""EdgeFlow Interactive Validation and CLI Tooling

This module provides interactive validation capabilities, incremental validation
during development, and enhanced CLI tooling with real-time feedback.
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from traceability_system import ProvenanceTracker, TransformationType, trace_transformation

logger = logging.getLogger(__name__)
console = Console()


class ValidationSeverity(Enum):
    """Validation message severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationMessage:
    """A validation message with context."""
    severity: ValidationSeverity
    code: str
    message: str
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggestion: Optional[str] = None
    fix_command: Optional[str] = None
    documentation_url: Optional[str] = None
    
    def __str__(self) -> str:
        location = ""
        if self.source_file:
            location = f"{self.source_file}"
            if self.line_number:
                location += f":{self.line_number}"
                if self.column_number:
                    location += f":{self.column_number}"
            location += " - "
        
        return f"[{self.severity.value.upper()}] {location}{self.message}"


@dataclass
class ValidationResult:
    """Results from validation process."""
    success: bool = True
    messages: List[ValidationMessage] = field(default_factory=list)
    validation_time_ms: float = 0.0
    files_validated: int = 0
    
    def has_errors(self) -> bool:
        """Check if there are any error messages."""
        return any(msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for msg in self.messages)
    
    def has_warnings(self) -> bool:
        """Check if there are any warning messages."""
        return any(msg.severity == ValidationSeverity.WARNING for msg in self.messages)
    
    def get_error_count(self) -> int:
        """Get number of error messages."""
        return sum(1 for msg in self.messages 
                  if msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
    
    def get_warning_count(self) -> int:
        """Get number of warning messages."""
        return sum(1 for msg in self.messages if msg.severity == ValidationSeverity.WARNING)


class InteractiveValidator:
    """Interactive validator with real-time feedback."""
    
    def __init__(self, tracker: Optional[ProvenanceTracker] = None):
        self.tracker = tracker or ProvenanceTracker()
        self.console = Console()
        self.validation_rules = {}
        self._load_validation_rules()
    
    def _load_validation_rules(self) -> None:
        """Load validation rules from configuration."""
        # This would typically load from a configuration file
        self.validation_rules = {
            "dsl_syntax": {
                "enabled": True,
                "severity": ValidationSeverity.ERROR,
                "description": "DSL syntax validation",
            },
            "parameter_ranges": {
                "enabled": True,
                "severity": ValidationSeverity.ERROR,
                "description": "Parameter range validation",
            },
            "device_compatibility": {
                "enabled": True,
                "severity": ValidationSeverity.WARNING,
                "description": "Device compatibility checks",
            },
            "performance_hints": {
                "enabled": True,
                "severity": ValidationSeverity.INFO,
                "description": "Performance optimization hints",
            },
        }
    
    def validate_file(
        self, 
        file_path: str, 
        target_device: Optional[str] = None,
        show_progress: bool = True,
    ) -> ValidationResult:
        """Validate a single EdgeFlow file."""
        start_time = time.perf_counter()
        result = ValidationResult()
        
        with trace_transformation(
            TransformationType.VALIDATION,
            "interactive_validator",
            f"Validating file: {file_path}",
            parameters={"file_path": file_path, "target_device": target_device},
        ) as ctx:
            
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Validating...", total=100)
                    
                    # Step 1: Syntax validation
                    progress.update(task, advance=20, description="Checking syntax...")
                    syntax_messages = self._validate_syntax(file_path)
                    result.messages.extend(syntax_messages)
                    
                    # Step 2: Parameter validation
                    progress.update(task, advance=30, description="Validating parameters...")
                    param_messages = self._validate_parameters(file_path)
                    result.messages.extend(param_messages)
                    
                    # Step 3: Device compatibility
                    if target_device:
                        progress.update(task, advance=25, description="Checking device compatibility...")
                        device_messages = self._validate_device_compatibility(file_path, target_device)
                        result.messages.extend(device_messages)
                    
                    # Step 4: Performance hints
                    progress.update(task, advance=25, description="Generating performance hints...")
                    perf_messages = self._generate_performance_hints(file_path, target_device)
                    result.messages.extend(perf_messages)
                    
                    progress.update(task, completed=100, description="Validation complete!")
            else:
                # Run validation without progress bar
                result.messages.extend(self._validate_syntax(file_path))
                result.messages.extend(self._validate_parameters(file_path))
                if target_device:
                    result.messages.extend(self._validate_device_compatibility(file_path, target_device))
                result.messages.extend(self._generate_performance_hints(file_path, target_device))
            
            result.success = not result.has_errors()
            result.files_validated = 1
            result.validation_time_ms = (time.perf_counter() - start_time) * 1000
            
            ctx.add_metric("validation_time_ms", result.validation_time_ms)
            ctx.add_metric("error_count", result.get_error_count())
            ctx.add_metric("warning_count", result.get_warning_count())
        
        return result
    
    def validate_directory(
        self, 
        directory_path: str,
        target_device: Optional[str] = None,
        recursive: bool = True,
    ) -> ValidationResult:
        """Validate all EdgeFlow files in a directory."""
        start_time = time.perf_counter()
        result = ValidationResult()
        
        # Find all .ef files
        ef_files = []
        path = Path(directory_path)
        if recursive:
            ef_files = list(path.rglob("*.ef"))
        else:
            ef_files = list(path.glob("*.ef"))
        
        if not ef_files:
            result.messages.append(ValidationMessage(
                ValidationSeverity.WARNING,
                "NO_FILES",
                f"No EdgeFlow (.ef) files found in {directory_path}",
                suggestion="Create .ef files or check the directory path",
            ))
            return result
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            main_task = progress.add_task("Validating directory...", total=len(ef_files))
            
            for ef_file in ef_files:
                progress.update(main_task, description=f"Validating {ef_file.name}...")
                
                file_result = self.validate_file(str(ef_file), target_device, show_progress=False)
                result.messages.extend(file_result.messages)
                result.files_validated += 1
                
                progress.advance(main_task)
        
        result.success = not result.has_errors()
        result.validation_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    def _validate_syntax(self, file_path: str) -> List[ValidationMessage]:
        """Validate DSL syntax."""
        messages = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Basic syntax checks
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check for assignment syntax
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) != 2:
                        messages.append(ValidationMessage(
                            ValidationSeverity.ERROR,
                            "SYNTAX_ERROR",
                            "Invalid assignment syntax",
                            source_file=file_path,
                            line_number=line_num,
                            suggestion="Use 'key = value' format",
                        ))
                    else:
                        key, value = parts[0].strip(), parts[1].strip()
                        if not key:
                            messages.append(ValidationMessage(
                                ValidationSeverity.ERROR,
                                "EMPTY_KEY",
                                "Empty parameter name",
                                source_file=file_path,
                                line_number=line_num,
                            ))
                        if not value:
                            messages.append(ValidationMessage(
                                ValidationSeverity.ERROR,
                                "EMPTY_VALUE",
                                "Empty parameter value",
                                source_file=file_path,
                                line_number=line_num,
                            ))
                else:
                    messages.append(ValidationMessage(
                        ValidationSeverity.WARNING,
                        "UNRECOGNIZED_LINE",
                        f"Unrecognized line format: {line}",
                        source_file=file_path,
                        line_number=line_num,
                        suggestion="Use 'key = value' format or add comment with #",
                    ))
        
        except Exception as e:
            messages.append(ValidationMessage(
                ValidationSeverity.ERROR,
                "FILE_READ_ERROR",
                f"Could not read file: {e}",
                source_file=file_path,
            ))
        
        return messages
    
    def _validate_parameters(self, file_path: str) -> List[ValidationMessage]:
        """Validate parameter values and ranges."""
        messages = []
        
        try:
            # Load and parse the file
            from parser import parse_ef
            config = parse_ef(file_path)
            
            # Validate quantization parameter
            quantize = config.get("quantize", "none")
            if quantize not in ["none", "int8", "float16", "dynamic"]:
                messages.append(ValidationMessage(
                    ValidationSeverity.ERROR,
                    "INVALID_QUANTIZATION",
                    f"Invalid quantization type: {quantize}",
                    source_file=file_path,
                    suggestion="Use one of: none, int8, float16, dynamic",
                ))
            
            # Validate target device
            target_device = config.get("target_device", "cpu")
            valid_devices = ["cpu", "gpu", "raspberry_pi", "edge", "mobile", "server"]
            if target_device not in valid_devices:
                messages.append(ValidationMessage(
                    ValidationSeverity.WARNING,
                    "UNKNOWN_DEVICE",
                    f"Unknown target device: {target_device}",
                    source_file=file_path,
                    suggestion=f"Consider using one of: {', '.join(valid_devices)}",
                ))
            
            # Validate optimization level
            optimize_for = config.get("optimize_for", "balanced")
            if optimize_for not in ["size", "speed", "balanced", "accuracy"]:
                messages.append(ValidationMessage(
                    ValidationSeverity.WARNING,
                    "INVALID_OPTIMIZATION",
                    f"Invalid optimization target: {optimize_for}",
                    source_file=file_path,
                    suggestion="Use one of: size, speed, balanced, accuracy",
                ))
        
        except Exception as e:
            messages.append(ValidationMessage(
                ValidationSeverity.ERROR,
                "PARSE_ERROR",
                f"Could not parse configuration: {e}",
                source_file=file_path,
            ))
        
        return messages
    
    def _validate_device_compatibility(self, file_path: str, target_device: str) -> List[ValidationMessage]:
        """Validate compatibility with target device."""
        messages = []
        
        try:
            from device_specs import get_device_specs
            device_specs = get_device_specs(target_device)
            
            if not device_specs:
                messages.append(ValidationMessage(
                    ValidationSeverity.WARNING,
                    "UNKNOWN_DEVICE_SPECS",
                    f"No specifications found for device: {target_device}",
                    source_file=file_path,
                ))
                return messages
            
            # Check memory constraints
            max_memory_mb = device_specs.get("max_memory_mb", float('inf'))
            if max_memory_mb < 100:  # Arbitrary threshold
                messages.append(ValidationMessage(
                    ValidationSeverity.WARNING,
                    "LOW_MEMORY_DEVICE",
                    f"Target device has limited memory: {max_memory_mb}MB",
                    source_file=file_path,
                    suggestion="Consider aggressive quantization and pruning",
                ))
            
            # Check compute constraints
            max_ops_per_sec = device_specs.get("max_ops_per_sec", float('inf'))
            if max_ops_per_sec < 1e9:  # 1 GOPS threshold
                messages.append(ValidationMessage(
                    ValidationSeverity.INFO,
                    "LIMITED_COMPUTE",
                    f"Target device has limited compute: {max_ops_per_sec/1e6:.0f} MOPS",
                    source_file=file_path,
                    suggestion="Consider model pruning and operator fusion",
                ))
        
        except Exception as e:
            messages.append(ValidationMessage(
                ValidationSeverity.WARNING,
                "DEVICE_CHECK_ERROR",
                f"Could not validate device compatibility: {e}",
                source_file=file_path,
            ))
        
        return messages
    
    def _generate_performance_hints(self, file_path: str, target_device: Optional[str]) -> List[ValidationMessage]:
        """Generate performance optimization hints."""
        messages = []
        
        try:
            from parser import parse_ef
            config = parse_ef(file_path)
            
            # Hint about quantization
            if config.get("quantize", "none") == "none" and target_device in ["raspberry_pi", "edge", "mobile"]:
                messages.append(ValidationMessage(
                    ValidationSeverity.INFO,
                    "QUANTIZATION_HINT",
                    "Consider enabling INT8 quantization for better performance on edge devices",
                    source_file=file_path,
                    suggestion="Add 'quantize = int8' to your configuration",
                    fix_command="echo 'quantize = int8' >> " + file_path,
                ))
            
            # Hint about pruning
            if "pruning" not in config and target_device in ["raspberry_pi", "edge"]:
                messages.append(ValidationMessage(
                    ValidationSeverity.INFO,
                    "PRUNING_HINT",
                    "Consider enabling model pruning to reduce model size",
                    source_file=file_path,
                    suggestion="Add 'enable_pruning = true' to your configuration",
                ))
            
            # Hint about operator fusion
            if config.get("enable_fusion", True) is False:
                messages.append(ValidationMessage(
                    ValidationSeverity.INFO,
                    "FUSION_HINT",
                    "Operator fusion is disabled, which may impact performance",
                    source_file=file_path,
                    suggestion="Consider enabling operator fusion for better performance",
                ))
        
        except Exception:
            # Don't add error messages for hint generation failures
            pass
        
        return messages
    
    def display_results(self, result: ValidationResult, verbose: bool = False) -> None:
        """Display validation results in a formatted way."""
        # Summary panel
        summary_text = Text()
        if result.success:
            summary_text.append("✅ Validation PASSED", style="bold green")
        else:
            summary_text.append("❌ Validation FAILED", style="bold red")
        
        summary_text.append(f"\n📁 Files validated: {result.files_validated}")
        summary_text.append(f"\n⏱️  Validation time: {result.validation_time_ms:.1f}ms")
        summary_text.append(f"\n🔍 Total messages: {len(result.messages)}")
        
        if result.has_errors():
            summary_text.append(f"\n❌ Errors: {result.get_error_count()}", style="red")
        if result.has_warnings():
            summary_text.append(f"\n⚠️  Warnings: {result.get_warning_count()}", style="yellow")
        
        self.console.print(Panel(summary_text, title="Validation Summary", border_style="blue"))
        
        # Messages table
        if result.messages and (verbose or result.has_errors()):
            table = Table(title="Validation Messages")
            table.add_column("Severity", style="bold")
            table.add_column("Code")
            table.add_column("Location")
            table.add_column("Message")
            table.add_column("Suggestion")
            
            for msg in result.messages:
                # Skip info messages unless verbose
                if not verbose and msg.severity == ValidationSeverity.INFO:
                    continue
                
                severity_style = {
                    ValidationSeverity.INFO: "blue",
                    ValidationSeverity.WARNING: "yellow",
                    ValidationSeverity.ERROR: "red",
                    ValidationSeverity.CRITICAL: "bold red",
                }[msg.severity]
                
                location = ""
                if msg.source_file:
                    location = Path(msg.source_file).name
                    if msg.line_number:
                        location += f":{msg.line_number}"
                
                table.add_row(
                    Text(msg.severity.value.upper(), style=severity_style),
                    msg.code,
                    location,
                    msg.message,
                    msg.suggestion or "",
                )
            
            self.console.print(table)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--device', '-d', help='Target device for validation')
@click.pass_context
def cli(ctx, verbose, device):
    """EdgeFlow Interactive Validation CLI."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['device'] = device
    ctx.obj['validator'] = InteractiveValidator()


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_context
def validate_file(ctx, file_path):
    """Validate a single EdgeFlow file."""
    validator = ctx.obj['validator']
    device = ctx.obj['device']
    verbose = ctx.obj['verbose']
    
    console.print(f"🔍 Validating file: [bold]{file_path}[/bold]")
    
    result = validator.validate_file(file_path, device)
    validator.display_results(result, verbose)
    
    # Exit with error code if validation failed
    if not result.success:
        sys.exit(1)


@cli.command()
@click.argument('directory_path', type=click.Path(exists=True, file_okay=False))
@click.option('--recursive', '-r', is_flag=True, help='Validate files recursively')
@click.pass_context
def validate_dir(ctx, directory_path, recursive):
    """Validate all EdgeFlow files in a directory."""
    validator = ctx.obj['validator']
    device = ctx.obj['device']
    verbose = ctx.obj['verbose']
    
    console.print(f"🔍 Validating directory: [bold]{directory_path}[/bold]")
    if recursive:
        console.print("📁 Recursive mode enabled")
    
    result = validator.validate_directory(directory_path, device, recursive)
    validator.display_results(result, verbose)
    
    # Exit with error code if validation failed
    if not result.success:
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--watch', '-w', is_flag=True, help='Watch file for changes')
@click.pass_context
def check(ctx, file_path, watch):
    """Quick syntax check of an EdgeFlow file."""
    validator = ctx.obj['validator']
    
    def check_file():
        result = validator.validate_file(file_path, show_progress=False)
        
        if result.success:
            console.print("✅ [green]Syntax OK[/green]")
        else:
            console.print("❌ [red]Syntax Errors Found[/red]")
            for msg in result.messages:
                if msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                    console.print(f"  {msg}")
        
        return result.success
    
    if watch:
        console.print(f"👀 Watching file: [bold]{file_path}[/bold] (Press Ctrl+C to stop)")
        
        import time
        from pathlib import Path
        
        last_modified = Path(file_path).stat().st_mtime
        
        try:
            while True:
                current_modified = Path(file_path).stat().st_mtime
                if current_modified > last_modified:
                    console.print("\n🔄 File changed, re-validating...")
                    check_file()
                    last_modified = current_modified
                
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n👋 Stopped watching file")
    else:
        success = check_file()
        if not success:
            sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_context
def explain(ctx, file_path):
    """Explain validation results with detailed suggestions."""
    validator = ctx.obj['validator']
    device = ctx.obj['device']
    
    console.print(f"🧠 Analyzing file: [bold]{file_path}[/bold]")
    
    result = validator.validate_file(file_path, device)
    
    # Show detailed explanations
    for msg in result.messages:
        panel_style = {
            ValidationSeverity.INFO: "blue",
            ValidationSeverity.WARNING: "yellow",
            ValidationSeverity.ERROR: "red",
            ValidationSeverity.CRITICAL: "bold red",
        }[msg.severity]
        
        content = Text()
        content.append(f"Code: {msg.code}\n", style="bold")
        content.append(f"Message: {msg.message}\n")
        
        if msg.suggestion:
            content.append(f"💡 Suggestion: {msg.suggestion}\n", style="cyan")
        
        if msg.fix_command:
            content.append(f"🔧 Quick fix: ", style="green")
            content.append(f"{msg.fix_command}\n", style="green bold")
        
        if msg.documentation_url:
            content.append(f"📚 Documentation: {msg.documentation_url}\n", style="blue")
        
        location = ""
        if msg.source_file:
            location = f" - {Path(msg.source_file).name}"
            if msg.line_number:
                location += f":{msg.line_number}"
        
        console.print(Panel(
            content,
            title=f"{msg.severity.value.upper()}{location}",
            border_style=panel_style,
        ))


if __name__ == "__main__":
    cli()
