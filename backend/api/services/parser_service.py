from __future__ import annotations

from parser import EdgeFlowParserError, parse_edgeflow_string, validate_config
from typing import Any, Dict, Tuple

from edgeflow_ast import create_program_from_dict
from edgeflow_ir import create_ir_from_config
from error_reporter import EdgeFlowErrorReporter
from static_validator import validate_edgeflow_config_static


class ParserService:
    """Service for parsing EdgeFlow configurations in API context."""

    @staticmethod
    def parse_config_content(content: str) -> Tuple[bool, Dict[str, Any], str]:
        """Parse EdgeFlow configuration content.

        Returns:
            Tuple of (success, config_dict, error_message)
        """
        try:
            config = parse_edgeflow_string(content)
            is_valid, errors = validate_config(config)
            if not is_valid:
                return False, {}, "; ".join(errors)
            return True, config, ""
        except (
            EdgeFlowParserError
        ) as e:  # pragma: no cover - exercised via tests elsewhere
            return False, {}, str(e)
        except Exception as e:  # noqa: BLE001
            return False, {}, f"Unexpected error: {str(e)}"

    @staticmethod
    def parse_config_with_diagnostics(content: str) -> Dict[str, Any]:
        """Parse content and return config, IR placeholder, and structured diagnostics.

        This method is additive and does not change existing behavior.
        """
        result: Dict[str, Any] = {
            "ok": False,
            "config": {},
            "diagnostics": [],
            "errors": "",
            "ast": None,
            "ir": None,
        }
        try:
            cfg = parse_edgeflow_string(content)
            result["config"] = cfg
            # Build AST from config dict
            try:
                program = create_program_from_dict(cfg)
                result["ast"] = program.to_dict()
            except Exception:
                result["ast"] = None
            # Run existing simple validation
            is_valid, errors = validate_config(cfg)
            if not is_valid:
                result["errors"] = "; ".join(errors)
            # Run static validator for richer issues
            static = validate_edgeflow_config_static(cfg)
            reporter = EdgeFlowErrorReporter()
            diags = []
            for issue in static.issues + static.warnings:
                rep = reporter.generate_error_report(issue)
                diags.append(
                    {
                        "id": rep.error_id,
                        "severity": rep.severity.value,
                        "category": rep.category.value,
                        "title": rep.title,
                        "message": rep.message,
                        "hint": rep.suggestions[0] if rep.suggestions else None,
                    }
                )
            result["diagnostics"] = diags
            # Build IR graph dictionary if parsing was okay
            try:
                graph = create_ir_from_config(cfg)
                result["ir"] = graph.to_dict()
            except Exception:
                result["ir"] = None
            result["ok"] = is_valid and static.is_valid
            return result
        except EdgeFlowParserError as e:
            result["errors"] = str(e)
            return result
        except Exception as e:  # noqa: BLE001
            result["errors"] = f"Unexpected error: {str(e)}"
            return result
