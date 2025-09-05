EdgeFlow Compiler (CLI)
=======================

EdgeFlow is a domain-specific language (DSL) and compiler for optimizing TensorFlow Lite models for edge deployment. Users write simple `.ef` configuration files that describe optimization strategies (e.g., INT8 quantization), target devices, and goals like latency or size. The CLI parses these configs and runs the optimization pipeline.

Project Status: Day 1 CLI skeleton with tests and CI. Parser and optimizer are stubbed for integration by other teams.

Overview
--------
- Language: EdgeFlow `.ef` configuration files
- Targets: TFLite models on edge devices (e.g., Raspberry Pi)
- Pipeline: parse config → optimize model → (future) benchmark → report

Example `.ef`
-------------
```
model_path = "path/to/model.tflite"
output_path = "path/to/optimized_model.tflite"
quantize = int8
target_device = "raspberry_pi"
optimize_for = latency
```

Installation
------------
- Python 3.8–3.11 supported (tested in CI)
- Install dependencies:
```
pip install -r requirements.txt
```

For development (linting, tests, coverage, hooks):
```
pip install -r requirements-dev.txt
```

Usage
-----
Basic:
```
python edgeflowc.py path/to/config.ef
```

Verbose:
```
python edgeflowc.py path/to/config.ef --verbose
```

Help and Version:
```
python edgeflowc.py --help
python edgeflowc.py --version
```

Expected Behavior
-----------------
- Missing file:
```
python edgeflowc.py non_existent.ef
# Error: File 'non_existent.ef' not found
```

- Wrong extension:
```
python edgeflowc.py invalid.txt
# Error: Invalid file extension. Expected '.ef' file
```

CLI Options
-----------
- `config_path`: Positional `.ef` file path (required)
- `-v, --verbose`: Enable verbose debug output
- `--version`: Print CLI version and exit

=======
- Python 3.8+
- Java JDK (required by ANTLR)
- `antlr4-python3-runtime` library (`pip install antlr4-python3-runtime`)
- TensorFlow and TensorFlow Lite Converter (`pip install tensorflow`)
- ANTLR 4.13.1 Complete Jar (download from [antlr.org](https://www.antlr.org/download/antlr-4.13.1-complete.jar) and place in `grammer/`)
- SSH access/setup for target device (if deploying physically)

### Clone the Repository

git clone <https://github.com/yourusername/edge-ai-dsl.git>
cd edge-ai-dsl

### Generate the Parser with ANTLR

java -jar grammer/antlr-4.13.1-complete.jar -Dlanguage=Python3 -o parser grammer/EdgeFlow.g4

### Running the DSL Compiler

Currently, you can parse DSL scripts and trigger basic quantization:

python main.py examples/sample.dsl

---

## Project Structure

Architecture
------------
```
edgeFlow/
├── edgeflowc.py          # CLI entry point (this repo)
├── parser.py             # ANTLR-based parser (Team A)
├── optimizer.py          # Model optimization logic (Team B)
├── benchmarker.py        # Performance measurement tools
├── reporter.py           # Report generation
├── tests/                # Unit tests
│   ├── __init__.py
│   └── test_cli.py
├── .github/workflows/ci.yml   # CI: lint, type, test, coverage badge
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Dev/test dependencies
├── README.md             # This file
└── .pre-commit-config.yaml    # Pre-commit hooks
```

Integration Points
------------------
- Parser (`parser.parse_ef(path)`): `edgeflowc.load_config` tries to import and call this. If not found yet, it falls back to returning a minimal config with raw text.
- Optimizer (`optimizer.optimize(config)`): `edgeflowc.optimize_model` tries to import and call this. If not found yet, it logs a message and continues.

Development
-----------
Set up pre-commit hooks:
```
pre-commit install
```

Run linters and type checks:
```
black .
isort --profile black .
flake8 .
mypy --ignore-missing-imports .
```

Run tests with coverage:
```
pytest -q --cov=edgeflowc --cov-report=term-missing
```

CI/CD
-----
GitHub Actions runs on pushes and PRs for Python 3.11:
- Lint: black, isort, flake8
- Type check: mypy (ignore missing imports by default)
- Tests with coverage (fail below 90%)
- Coverage badge artifact generated via `genbadge`

Contributing
------------
- Open a PR with a focused set of changes
- Ensure `black`, `isort`, `flake8`, and `mypy` pass
- Add/Update tests to maintain ≥90% coverage
- Clearly document changes in docstrings and README where relevant

Security Notes
--------------
- The CLI validates that the input path is a regular file with a `.ef` extension.
- Paths are normalized and resolved; the CLI does not follow any network or remote sources.
- Future work: sandbox model handling and ensure safe file operations during optimization.

License
-------
TBD (add the appropriate license file for your project).
