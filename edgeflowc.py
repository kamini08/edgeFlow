"""EdgeFlow Compiler CLI.

This module implements the command-line interface (CLI) skeleton for the
EdgeFlow compiler. It parses EdgeFlow configuration files (``.ef``) and
coordinates the optimization pipeline by delegating to the parser and
optimizer modules.

Day 1 focuses on a robust, testable CLI with placeholders for integration.

Example:
    $ python edgeflowc.py model_config.ef --verbose

"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from ast import Dict
from parser import parse_ef  # Backward-compatible name
from typing import Any, Dict as DictType

# Import our modules
from optimizer import optimize

try:  # Prefer Day 2 API if present
    from parser import parse_edgeflow_file as _parse_edgeflow_file  # type: ignore
except Exception:  # noqa: BLE001
    _parse_edgeflow_file = None  # type: ignore

from code_generator import CodeGenerator, generate_code
from edgeflow_ast import create_program_from_dict
from reporter import generate_report
from validator import validate_edgeflow_config, validate_model_compatibility

VERSION = "0.1.0"


def _configure_logging(verbose: bool) -> None:
    """Configure root logger.

    Args:
        verbose: Whether to enable verbose (DEBUG) logging.
    """

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog="edgeflowc",
        description=(
            "EdgeFlow compiler for optimizing TFLite models using DSL configs"
        ),
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Path to EdgeFlow configuration file (.ef)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"edgeflowc {VERSION}",
        help="Show version and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate config, print result, exit",
    )

    args = parser.parse_args()
    return args


def validate_file_path(file_path: str) -> bool:
    """Validate that the input file exists and has correct extension.

    Ensures the provided path resolves to an existing regular file and has
    a case-insensitive ``.ef`` extension.

    Args:
        file_path: Path to the EdgeFlow configuration file.

    Returns:
        bool: True if the path is valid, otherwise False.
    """

    if not file_path:
        return False

    try:
        # Normalize and resolve the path to avoid oddities.
        normalized = os.path.normpath(file_path)
        # Abspath is sufficient for a local CLI to avoid relative confusion.
        abs_path = os.path.abspath(normalized)
    except Exception:
        return False

    if not os.path.isfile(abs_path):
        return False

    _, ext = os.path.splitext(abs_path)
    return ext.lower() == ".ef"


def _load_project_parser_module():
    """Load the project's parser module safely despite stdlib name conflict.

    Returns a module-like object that may expose Day 2 APIs
    (parse_edgeflow_file, validate_config) or Day 1 API (parse_ef).
    Prefers any test-provided sys.modules['parser'] to preserve monkeypatching.
    """

    if "parser" in sys.modules:
        return sys.modules["parser"]

    # Attempt to load package 'parser' from the repo (parser/__init__.py)
    try:
        import os

        root = os.path.abspath(os.path.dirname(__file__))
        pkg_init = os.path.join(root, "parser", "__init__.py")
        if os.path.isfile(pkg_init):
            spec = importlib.util.spec_from_file_location(
                "edgeflow_project_parser", pkg_init
            )
            if spec and spec.loader:  # type: ignore[truthy-bool]
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[arg-type]
                return mod
    except Exception:
        pass

    # As a last resort, try loading top-level parser.py next to this file
    try:
        import os

        mod_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "parser.py")
        if os.path.isfile(mod_path):
            spec = importlib.util.spec_from_file_location(
                "edgeflow_parser_core", mod_path
            )
            if spec and spec.loader:  # type: ignore[truthy-bool]
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[arg-type]
                return mod
    except Exception:
        pass

    return None


def load_config(file_path: str) -> DictType[str, Any]:
    """Load and validate EdgeFlow configuration from file.

    Args:
        file_path: Path to the ``.ef`` configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    try:
        # Prefer modern parser API if available
        if _parse_edgeflow_file is not None:
            config = _parse_edgeflow_file(file_path)
        else:
            config = parse_ef(file_path)

        # Comprehensive semantic validation
        is_valid, errors = validate_edgeflow_config(config)
        if not is_valid:
            logging.error("Configuration validation failed:")
            for error in errors:
                logging.error(f"  - {error}")
            raise SystemExit(1)

        # Model compatibility validation
        model_path = config.get("model")
        if model_path:
            is_compatible, warnings = validate_model_compatibility(model_path, config)
            if warnings:
                logging.warning("Model compatibility warnings:")
                for warning in warnings:
                    logging.warning(f"  - {warning}")

        logging.info("Configuration validation passed")
        return config

    except Exception as exc:
        logging.error("Failed to load configuration: %s", exc)
        raise SystemExit(1)


def optimize_model(config: DictType[str, Any]) -> DictType[str, Any]:
    """Run the full Day 3/4 pipeline: benchmark -> optimize -> benchmark.

    This reorders the earlier logic so we always capture baseline metrics
    prior to optimization (as required by Phase II tasks). It then performs
    optimization and captures post-optimization metrics plus a comparison.
    """
    try:
        from benchmarker import benchmark_model, compare_models
        from optimizer import optimize

        model_path = config.get("model", "model.tflite")
        if not os.path.exists(model_path):
            logging.warning(
                "Model file not found: %s (a test model may be generated by optimizer)",
                model_path,
            )

        logging.info("=== BASELINE BENCHMARK (Pre-Optimization) ===")
        original_benchmark = benchmark_model(model_path, config)

        logging.info("=== OPTIMIZATION PHASE ===")
        optimized_path, opt_results = optimize(config)

        logging.info("=== POST-OPTIMIZATION BENCHMARK ===")
        optimized_benchmark = benchmark_model(optimized_path, config)

        logging.info("=== COMPARISON ===")
        comparison = compare_models(model_path, optimized_path, config)
        improvements = comparison.get("improvements", {})

        logging.info("=== EDGEFLOW OPTIMIZATION SUMMARY ===")
        logging.info(
            "Model size reduction: %.1f%%",
            improvements.get("size_reduction_percent", 0.0),
        )
        logging.info(
            "Latency improvement: %.1f%%",
            improvements.get("latency_improvement_percent", 0.0),
        )
        logging.info(
            "Throughput improvement: %.1f%%",
            improvements.get("throughput_improvement_percent", 0.0),
        )
        logging.info(
            "Memory improvement: %.1f%%",
            improvements.get("memory_improvement_percent", 0.0),
        )
        logging.info("Optimized model saved to: %s", optimized_path)

        return {
            "optimization": opt_results,
            "original_benchmark": original_benchmark,
            "optimized_benchmark": optimized_benchmark,
            "comparison": comparison,
        }
    except Exception as e:  # noqa: BLE001
        logging.error("Optimization pipeline failed: %s", e)
        return {"error": str(e)}


def main() -> int:
    """Main entry point for EdgeFlow compiler.

    Returns:
        int: Process exit code (0 on success, non-zero on error).
    """

    try:
        args = parse_arguments()
        _configure_logging(args.verbose)

        if not args.config_path:
            logging.error("No configuration file provided. See --help.")
            return 2

        if not validate_file_path(args.config_path):
            # Provide a specific error where possible.
            if not os.path.exists(args.config_path):
                logging.error("Error: File '%s' not found", args.config_path)
            else:
                logging.error("Error: Invalid file extension. Expected '.ef' file")
            return 1

        # Parse configuration file
        cfg = load_config(args.config_path)
        if getattr(args, "dry_run", False):
            # Print parsed config to stdout and exit without optimization
            print(json.dumps(cfg, indent=2))
            logging.info("Configuration parsed successfully (dry-run)")
            return 0
        logging.debug("Loaded config: %s", json.dumps(cfg, indent=2)[:500])

        # Create AST from parsed configuration
        program = create_program_from_dict(cfg)
        logging.info("Created AST with %d statements", len(program.statements))

        # Generate code
        logging.info("Generating inference code...")
        generator = CodeGenerator(program)

        # Generate Python code
        python_code = generator.generate_python_inference()
        logging.info(
            "Generated Python inference code (%d characters)", len(python_code)
        )

        # Generate C++ code
        cpp_code = generator.generate_cpp_inference()
        logging.info("Generated C++ inference code (%d characters)", len(cpp_code))

        # Generate optimization report
        report = generator.generate_optimization_report()
        logging.info("Generated optimization report (%d characters)", len(report))

        # Save generated files
        output_dir = "generated"
        files = generate_code(program, output_dir)
        logging.info("Saved generated files to %s:", output_dir)
        for file_type, file_path in files.items():
            logging.info("  %s: %s", file_type, file_path)

        # Run optimization pipeline
        logging.info("Running EdgeFlow optimization pipeline...")
        opt_results = optimize_model(cfg)

        if "error" in opt_results:
            logging.error(f"Optimization failed: {opt_results['error']}")
            return 1

        # Generate human-readable optimization report using reporter module
        try:
            logging.info("\n📊 Generating optimization report...")
            original = opt_results.get("original_benchmark", {})
            optimized = opt_results.get("optimized_benchmark", {})

            unoptimized_stats = {
                "size_mb": float(original.get("model_size_mb", 0.0)),
                "latency_ms": float(original.get("latency_ms", 0.0)),
                "model_path": original.get(
                    "model_path", cfg.get("model", "model.tflite")
                ),
            }
            optimized_stats = {
                "size_mb": float(optimized.get("model_size_mb", 0.0)),
                "latency_ms": float(optimized.get("latency_ms", 0.0)),
                "model_path": optimized.get(
                    "model_path", cfg.get("model", "model.tflite")
                ),
            }

            report_path = generate_report(
                unoptimized_stats,
                optimized_stats,
                cfg,
                output_path="report.md",
            )
            logging.info("✅ Report generated: %s", report_path)

            # Optional concise summary
            size_red = (
                (
                    1
                    - (optimized_stats["size_mb"] or 0.0)
                    / (unoptimized_stats["size_mb"] or 1.0)
                )
                * 100.0
                if unoptimized_stats["size_mb"]
                else 0.0
            )
            speedup = (
                (unoptimized_stats["latency_ms"] or 0.0)
                / (optimized_stats["latency_ms"] or 1.0)
                if optimized_stats["latency_ms"]
                else 0.0
            )
            logging.info("=== Optimization Summary ===")
            logging.info("Size reduced by: %.1f%%", size_red)
            logging.info("Speed improved by: %.1fx", speedup)
        except Exception as e:  # noqa: BLE001
            logging.error("❌ Failed to generate report: %s", e)
            logging.debug("Report generation exception", exc_info=True)

        logging.info("EdgeFlow compilation pipeline completed successfully!")
        logging.info(
            "🎉 EdgeFlow has successfully optimized your model for edge deployment!"
        )
        return 0
    except SystemExit as e:
        # Argparse uses SystemExit for --help/--version and parse errors.
        # Propagate code to the caller.
        return int(e.code) if e.code is not None else 0
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logging.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised via tests calling main
    raise SystemExit(main())
