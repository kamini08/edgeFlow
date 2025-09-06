# EdgeFlow Parser Documentation

## Overview
The EdgeFlow parser transforms `.ef` configuration files into Python dictionaries using ANTLR-generated lexer and parser when available, with a robust Python fallback to ensure reliability in all environments.

## Language Syntax

### Basic Assignment
key = value

### Supported Data Types
- String: "text" or 'text'
- Integer: 123, -456
- Float: 3.14, -2.5, 1e-3
- Boolean: true, false
- Identifier: int8, raspberry_pi, latency

### Comments
Lines starting with `#` are ignored. Inline comments after a value are supported:

This is a comment
model_path = "model.tflite"  # Inline comment

## API Reference

### `parse_edgeflow_file(file_path: str) -> Dict[str, Any]`
Parse an EdgeFlow configuration file.

Parameters:
- file_path: Path to .ef file

Returns:
- Dictionary with parsed configuration

Raises:
- EdgeFlowParserError: On syntax errors
- FileNotFoundError: If file doesn't exist

### `parse_edgeflow_string(content: str) -> Dict[str, Any]`
Parse EdgeFlow configuration from string.

Parameters:
- content: Configuration as string

Returns:
- Dictionary with parsed configuration

### `validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]`
Validate semantic constraints (required fields and ranges).

## Error Handling

The parser provides detailed error messages:

try:
    config = parse_edgeflow_file("config.ef")
except EdgeFlowParserError as e:
    print(f"Parse error: {e}")
    # Output: Parse error: Line 3: syntax error - expected single '=' in assignment

## Integration Examples

CLI Integration

from parser import parse_edgeflow_file

config = parse_edgeflow_file("model_config.ef")
print(f"Model: {config['model_path']}")

API Integration

from parser import parse_edgeflow_string

@app.post("/compile")
async def compile(content: str):
    config = parse_edgeflow_string(content)
    return {"config": config}

## Grammar Specification
See `grammer/EdgeFlow.g4` for the ANTLR grammar; when generated, artifacts are placed under the `parser/` package.

## PR Checklist

Before raising the PR, ensure:

- Run `./scripts/pre-commit-parser.sh` - ALL checks MUST pass
- Parser module has >90% test coverage
- All integration tests pass (CLI, API, Frontend)
- Documentation is complete
- ANTLR files are properly generated (if using ANTLR path)
- Error messages are user-friendly
- Type hints are complete
- No lint violations (flake8, mypy, black)
- CI workflow passes on all Python versions

## PR Description Template

### Parser Module Implementation - Day 2 Team A

Changes
- Created `parser.py` module with EdgeFlowConfigVisitor and fallback
- Implemented `parse_edgeflow_file()` and `parse_edgeflow_string()`
- Added comprehensive error handling
- Integrated with CLI (`edgeflowc.py`) including `--dry-run`
- Integrated with Backend API using `ParserService`
- Added validation logic

Testing
- Unit tests: >90% coverage on parser
- Integration tests: CLI and API (optional)
- All linters pass
- CI/CD workflow configured

Evidence

$ ./scripts/pre-commit-parser.sh
🔍 Running parser module pre-commit checks...
📝 Formatting code with black...
🧹 Running linters...
🧪 Running parser tests...
📊 Checking coverage...
🔗 Running integration tests...
💨 Running CLI smoke test...
✅ All parser checks passed!

## Important Notes

- The parser is the bridge between user input and system processing and must be robust
- Error messages should guide users to fix their configuration
- Parsing should be fast even for large configs
- The parser is extensible for future language features
- Maintain backward compatibility if grammar changes
- Consider adding a `--validate` flag to CLI for config validation without execution

