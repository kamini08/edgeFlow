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
- Python 3.11 (CI target)
- Install runtime dependencies:
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

Language Toolchain (ANTLR)
-------------------------
Prereqs:
- Java JDK (required by ANTLR tool)
- `antlr4-python3-runtime` (`pip install antlr4-python3-runtime`)
- ANTLR 4.13.1 Complete Jar (download from antlr.org and place in `grammer/`)

Generate Python parser/lexer into the `parser/` package:
```
java -jar grammer/antlr-4.13.1-complete.jar -Dlanguage=Python3 -o parser grammer/EdgeFlow.g4
```
After generation, `parser/` contains `EdgeFlowLexer.py`, `EdgeFlowParser.py`, `EdgeFlowVisitor.py`, etc. The CLI automatically uses them when present; otherwise it falls back to a simple line-based parser.

Running the Compiler
--------------------
Parse a `.ef` config and run the (placeholder) optimization pipeline:
```
python edgeflowc.py path/to/config.ef
```

## Project Structure

Architecture
------------
```
edgeFlow/
├── edgeflowc.py          # CLI entry point (this repo)
├── parser/               # ANTLR-generated modules + wrapper (__init__.py)
├── optimizer.py          # Model optimization logic (Team B)
├── benchmarker.py        # Performance measurement tools
├── reporter.py           # Report generation
├── tests/                # Unit tests
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

Web Interface
-------------
Backend (FastAPI):
- App entry: `backend/app.py`
- Endpoints with strict CLI parity:
  - `POST /api/compile` (maps to `python edgeflowc.py config.ef`)
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
```
docker-compose up --build
# Backend: http://localhost:8000/docs
# Frontend: http://localhost:3000
```

Production (CD + Reverse Proxy)
-------------------------------
- Continuous Deployment builds/pushes GHCR images, then deploys over SSH with Docker Compose on the server.
- Public site: https://edgeflow.pointblank.club/
- Host ports by default:
  - Backend: `18000` (container 8000)
  - Frontend: `13000` (container 3000)
- Recommended: bind services to `127.0.0.1` and expose via Nginx with TLS (Certbot). Frontend proxies `/api/*` to backend inside the Docker network; backend need not be directly exposed.

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
