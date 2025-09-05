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
import json
import logging
import os
import sys
from typing import Any, Dict


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

    # Try using the dedicated parser if present.
    try:
        from parser import parse_ef  # type: ignore

        return parse_ef(file_path)
    except Exception:  # noqa: BLE001 - broad until parser is implemented
        # Fallback: best-effort minimal config to enable end-to-end flow.
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
        return {
            "__source__": os.path.abspath(file_path),
            "__raw__": raw,
        }


def optimize_model(config: Dict[str, Any]) -> None:
    """Placeholder for model optimization pipeline.

    Attempts to locate and call the project's optimizer. If not available,
    logs a message and returns.

    Args:
        config: Parsed configuration dictionary produced by ``load_config``.
    """

    try:
        from optimizer import optimize  # type: ignore

        optimize(config)
    except Exception:  # noqa: BLE001 - broad until optimizer is implemented
        logging.info("Optimizer not available; skipping optimization step.")


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
                logging.error(
                    "Error: Invalid file extension. Expected '.ef' file"
                )
            return 1

        cfg = load_config(args.config_path)
        logging.debug("Loaded config: %s", json.dumps(cfg, indent=2)[:500])
        optimize_model(cfg)
        logging.info("EdgeFlow compilation pipeline completed.")
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

