# EdgeFlow Compiler with Semantic Analysis

EdgeFlow is a domain-specific language (DSL) and compiler for optimizing TensorFlow Lite models for edge deployment. Users write simple `.ef` configuration files that describe optimization strategies (e.g., INT8 quantization), target devices, and goals like latency or size. The CLI parses these configs and runs the optimization pipeline.

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

## CLI Options

- `config_path`: Positional `.ef` file path (required)
- `-v, --verbose`: Enable verbose debug output
- `--version`: Print CLI version and exit

## Language Toolchain (ANTLR)

Prereqs:

- Java JDK (required by ANTLR tool)
- `antlr4-python3-runtime` (`pip install antlr4-python3-runtime`)
- ANTLR 4.13.1 Complete Jar (download from antlr.org and place in `grammer/`)

Generate Python parser/lexer into the `parser/` package:

```bash


```

After generation, `parser/` contains `EdgeFlowLexer.py`, `EdgeFlowParser.py`, `EdgeFlowVisitor.py`, etc. The CLI automatically uses them when present; otherwise it falls back to a simple line-based parser.

## Running the Compiler

Parse a `.ef` config and run the (placeholder) optimization pipeline:

```bash
python -m edgeflow.compiler.edgeflowc path/to/config.ef
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
├── src/
│   └── edgeflow/
│       ├── compiler/         # Core compiler logic
│       ├── deployment/       # Deployment orchestration
│       ├── optimization/     # Optimization strategies
│       ├── analysis/         # Static and semantic analysis
│       ├── reporting/        # Error reporting and explainability
│       ├── benchmarking/     # Performance benchmarking
│       ├── config/           # Configuration handling
│       ├── ir/               # Intermediate Representation
│       ├── pipeline/         # Pipeline orchestration
│       ├── utils/            # Utility functions
│       ├── parser/           # Parser logic
│       ├── backend/          # Backend API
│       └── frontend/         # Frontend application
├── scripts/              # Helper scripts
├── tests/                # Unit and integration tests
├── docs/                 # Documentation
├── .github/workflows/    # CI/CD workflows
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Dev/test dependencies
├── README.md             # This file
└── Makefile              # Build automation
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

## CI/CD

GitHub Actions runs on pushes and PRs for Python 3.11:

- Lint: black, isort, flake8
- Type check: mypy (ignore missing imports by default)
- Tests with coverage (fail below 90%)
- Coverage badge artifact generated via `genbadge`

## Web Interface

Backend (FastAPI):

- App entry: `src/edgeflow/backend/app.py`
- Endpoints with strict CLI parity:
  - `POST /api/compile` (maps to `python -m edgeflow.compiler.edgeflowc config.ef`)
  - `POST /api/compile/verbose` (maps to `--verbose`)
  - `POST /api/optimize` (optimization phase)
  - `POST /api/benchmark` (benchmarking)
  - `GET /api/version` (maps to `--version`)
  - `GET /api/help` (maps to `--help`)
  - `GET /api/health` (health check)

Frontend (Next.js + TS):

- Components under `frontend/src/components` and pages under `frontend/src/pages`
- API client in `frontend/src/services/api.ts`
- Styling via Tailwind CSS (see `frontend/src/styles/globals.css`)

Local run (Docker):

```bash
docker-compose up --build
# Backend: http://localhost:8000/docs
# Frontend: http://localhost:3000
```

## Production (CD + Reverse Proxy)

- Continuous Deployment builds/pushes GHCR images, then deploys over SSH with Docker Compose on the server.
- Public site: <https://edgeflow.pointblank.club/>
- Host ports by default:
  - Backend: `18000` (container 8000)
  - Frontend: `13000` (container 3000)
- Recommended: bind services to `127.0.0.1` and expose via Nginx with TLS (Certbot). Frontend proxies `/api/*` to backend inside the Docker network; backend need not be directly exposed.

## Contributing

- Open a PR with a focused set of changes
- Ensure `black`, `isort`, `flake8`, and `mypy` pass
- Add/Update tests to maintain ≥90% coverage
- Clearly document changes in docstrings and README where relevant

## Security Notes

- The CLI validates that the input path is a regular file with a `.ef` extension.
- Paths are normalized and resolved; the CLI does not follow any network or remote sources.
- Future work: sandbox model handling and ensure safe file operations during optimization.

## License

TBD (add the appropriate license file for your project).

## Compatibility Check (CLI)

- `--check-only`: run device compatibility check and exit
- `--device-spec-file <path>`: load custom device specs (CSV/JSON)
- `--skip-check`: skip the initial compatibility gate

See `docs/initial_check.md` for usage and API examples.
