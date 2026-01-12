# EdgeFlow Compiler with Semantic Analysis

**DSL for deploying AI models on edge devices.**

EdgeFlow is a domain-specific language (DSL) and compiler for optimizing AI models for edge deployment. Users write simple `.ef` configuration files that describe optimization strategies (e.g., INT8 quantization), target devices, and goals like latency or size. The CLI parses these configs and runs the optimization pipeline.

**NEW: Comprehensive Semantic Analysis System** - The compiler now includes a sophisticated semantic analyzer that validates DSL models for shape compatibility, parameter ranges, resource constraints, and device compatibility before code generation.

Project Status: CLI with semantic analysis, tests, and CI. Parser and optimizer integration points ready.

## Overview

- Language: EdgeFlow `.ef` configuration files
- Targets: TFLite models on edge devices (e.g., Raspberry Pi)
- Pipeline: parse config → optimize model → (future) benchmark → report

## Example `.ef`

```bash
model_path = "path/to/model.tflite"
output_path = "path/to/optimized_model.tflite"
quantize = int8
target_device = "raspberry_pi"
optimize_for = latency
```

## Installation

- Python 3.11 (CI target)
- Install runtime dependencies:

```bash
pip install -r requirements.txt
```

For development (linting, tests, coverage, hooks):

```bash
pip install -r requirements-dev.txt
```

**Recommended: Install as editable package**

```bash
pip install -e .
```

This allows you to use `edgeflow` command directly instead of `python -m edgeflow.compiler.edgeflowc`.

## Quick Start

### 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/pointblank-club/edgeFlow.git
cd edgeFlow

# Create and activate virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# On Windows: .venv\Scripts\activate

# Install package
pip install -e .
```

### 2. Verify Installation

```bash
edgeflow --version
edgeflow --help
```

> **Note**: You may see a warning about missing 'dynamic_device_profiles'. This is expected and doesn't affect functionality.

### 3. Run Example

```bash
# View the example configuration
cat examples/quick_start.ef

# Run the compiler (note: requires a .tflite model file)
edgeflow examples/quick_start.ef
```

### 4. Create Your Own Config

Create a file `my_config.ef`:

```
model = "path/to/your/model.tflite"
output = "optimized_model.tflite"
quantize = int8
target_device = "raspberry_pi"
optimize_for = latency
```

Then run:

```bash
edgeflow my_config.ef
```

### Alternative: Without Installation

If you prefer not to install, set `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd)/src
python -m edgeflow.compiler.edgeflowc examples/quick_start.ef
```

## Usage

Basic:

```bash
python -m edgeflow.compiler.edgeflowc path/to/config.ef
```

Verbose:

```bash
python -m edgeflow.compiler.edgeflowc path/to/config.ef --verbose
```

Help and Version:

```bash
python -m edgeflow.compiler.edgeflowc --help
python -m edgeflow.compiler.edgeflowc --version
```

## Expected Behavior

- Missing file:

```bash
python -m edgeflow.compiler.edgeflowc non_existent.ef
# Error: File 'non_existent.ef' not found
```

- Wrong extension:

```bash
python -m edgeflow.compiler.edgeflowc invalid.txt
# Error: Invalid file extension. Expected '.ef' file
```

## Semantic Analysis System

The EdgeFlow compiler now includes a comprehensive semantic analysis system that validates DSL models before code generation. This ensures that generated models are correct, efficient, and compatible with target devices.

### Key Features

- **Shape Compatibility Validation**: Ensures tensor shapes match between connected layers
- **Parameter Range Checking**: Validates that all layer parameters are within acceptable ranges
- **Device Compatibility**: Checks if the model is compatible with target device constraints
- **Resource Analysis**: Validates memory usage and computational requirements
- **Forbidden Configuration Detection**: Identifies problematic layer sequences and configurations
- **Graph Structure Validation**: Detects cycles, connectivity issues, and missing components

### Quick Start with Semantic Analysis

```python
from semantic_analyzer import SemanticAnalyzer, IRGraph, semantic_check
from semantic_analyzer import get_edge_device_config

# Create or load your IR graph
graph = create_your_model_graph()

# Run semantic analysis
config = get_edge_device_config()  # For edge devices
errors = semantic_check(graph, config)

# Check results
if errors.has_errors():
    errors.print_summary()
else:
    print("✅ Model validation passed!")
```

### Example Error Output

```
📊 Semantic Analysis Summary:
   Errors: 2
   Warnings: 1
   Info: 0
   Fatal: 0

📝 Detailed Report:
  [ERROR] at model.dsl:line 7: Expected input shape (1, 256), got (1, 28, 28, 3).
    Suggestion: Ensure the previous layer outputs shape (1, 256)
  [ERROR] at model.dsl:line 10: Dense layer requires Flatten layer after Conv2D
    Suggestion: Add a Flatten layer between the convolutional and dense layers
  [WARNING] at model.dsl:line 5: Kernel size 13 exceeds recommended maximum (11) for target device
```

## Project Structure

```bash
edgeFlow/
├── .github/workflows/    # CI/CD GitHub Actions
├── scripts/              # Build and verification scripts
├── src/edgeflow/         # Core source code
│   ├── analysis/         # Static and semantic analysis
│   ├── backend/          # Backend API
│   ├── benchmarking/     # Performance benchmarking
│   ├── compiler/         # Core compiler logic
│   ├── config/           # Configuration handling
│   ├── deployment/       # Deployment orchestration
│   ├── frontend/         # Frontend application
│   ├── ir/               # Intermediate Representation
│   ├── optimization/     # Optimization strategies
│   ├── parser/           # Parser logic
│   ├── pipeline/         # Pipeline orchestration
│   ├── reporting/        # Error reporting
│   ├── semantic_analyzer/ # Semantic analysis system
│   └── utils/            # Utility functions
├── tests/                # Unit and integration tests
├── Dockerfile            # Docker container build
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Dev/test dependencies
└── README.md             # This file
```

## Integration Points

- Parser (`parser.parse_ef(path)`): `edgeflowc.load_config` tries to import and call this. If not found yet, it falls back to returning a minimal config with raw text.
- Optimizer (`optimizer.optimize(config)`): `edgeflowc.optimize_model` tries to import and call this. If not found yet, it logs a message and continues.

## Development

Set up pre-commit hooks:

```bash
pre-commit install
```

Run linters and type checks:

```bash
black .
isort --profile black .
flake8 .
mypy --ignore-missing-imports .
```

Run tests with coverage:

```bash
pytest -q --cov=edgeflowc --cov-report=term-missing
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute to EdgeFlow.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security Notes
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
