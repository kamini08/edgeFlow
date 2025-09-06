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
from typing import Any, Dict

# Import our modules
from parser import parse_ef
from edgeflow_ast import create_program_from_dict
from code_generator import CodeGenerator, generate_code

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


def load_config(file_path: str) -> Dict[str, Any]:
    """Placeholder for loading and parsing EdgeFlow config.

    Attempts to integrate with the project parser. If no parser is available,
    returns a minimal configuration that contains the raw text and path to
    unblock the pipeline during early development.

    Args:
        file_path: Path to the ``.ef`` configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """

    # Prefer Day 2 parser API if available via local module loader.
    mod = _load_project_parser_module()
    if mod is not None:
        try:
            if hasattr(mod, "parse_edgeflow_file") and hasattr(mod, "validate_config"):
                cfg = mod.parse_edgeflow_file(file_path)  # type: ignore[attr-defined]
                is_valid, errors = mod.validate_config(cfg)  # type: ignore
                if not is_valid:
                    logging.error("Configuration validation failed:")
                    for err in errors:
                        logging.error("  - %s", err)
                    raise SystemExit(1)
                # Add metadata keys that tests expect
                with open(file_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                cfg["__source__"] = os.path.abspath(file_path)
                cfg["__raw__"] = raw
                return cfg
        except SystemExit:
            raise
        except Exception as exc:
            logging.debug("Day 2 parser failed (%s); trying Day 1 API", exc)

        # Back-compat: try Day 1 API if present
        try:
            if hasattr(mod, "parse_ef"):
                return mod.parse_ef(file_path)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Fallback: best-effort minimal config to enable end-to-end flow.
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()
    return {
        "__source__": os.path.abspath(file_path),
        "__raw__": raw,
    }


def optimize_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete EdgeFlow optimization pipeline.

    Args:
        config: Parsed configuration dictionary produced by ``load_config``.
        
    Returns:
        Dictionary with optimization results
    """
    try:
        from optimizer import optimize
        from benchmarker import benchmark_model, compare_models
        
        # Get model path
        model_path = config.get('model', 'model.tflite')
        if not os.path.exists(model_path):
            logging.warning(f"Model file not found: {model_path}, creating dummy model")
        
        # Run optimization
        logging.info("Starting EdgeFlow optimization pipeline...")
        optimized_path, opt_results = optimize(config)
        
        # Benchmark original model
        logging.info("Benchmarking original model...")
        original_benchmark = benchmark_model(model_path, config)
        
        # Benchmark optimized model
        logging.info("Benchmarking optimized model...")
        optimized_benchmark = benchmark_model(optimized_path, config)
        
        # Compare models
        logging.info("Comparing models...")
        comparison = compare_models(model_path, optimized_path, config)
        
        # Combine results
        results = {
            'optimization': opt_results,
            'original_benchmark': original_benchmark,
            'optimized_benchmark': optimized_benchmark,
            'comparison': comparison
        }
        
        # Print summary
        improvements = comparison.get('improvements', {})
        logging.info("=== EDGEFLOW OPTIMIZATION SUMMARY ===")
        logging.info(f"Model size reduction: {improvements.get('size_reduction_percent', 0):.1f}%")
        logging.info(f"Latency improvement: {improvements.get('latency_improvement_percent', 0):.1f}%")
        logging.info(f"Throughput improvement: {improvements.get('throughput_improvement_percent', 0):.1f}%")
        logging.info(f"Memory improvement: {improvements.get('memory_improvement_percent', 0):.1f}%")
        logging.info(f"Optimized model saved to: {optimized_path}")
        
        return results
        
    except Exception as e:  # noqa: BLE001
        logging.error(f"Optimization failed: {e}")
        return {'error': str(e)}


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
        logging.info("Generated Python inference code (%d characters)", len(python_code))
        
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
        
        if 'error' in opt_results:
            logging.error(f"Optimization failed: {opt_results['error']}")
            return 1
        
        logging.info("EdgeFlow compilation pipeline completed successfully!")
        logging.info("🎉 EdgeFlow has successfully optimized your model for edge deployment!")
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
