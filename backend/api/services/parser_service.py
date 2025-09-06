from __future__ import annotations

from parser import EdgeFlowParserError, parse_edgeflow_string, validate_config
from typing import Any, Dict, Tuple


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
