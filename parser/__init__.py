"""EdgeFlow DSL parser interface.

Exposes :func:`parse_ef` which attempts to use ANTLR-generated modules
(`EdgeFlowLexer`, `EdgeFlowParser`, `EdgeFlowVisitor`) if present under this
package. If they are not available, it falls back to a simple line-based parse
that supports ``key = value`` pairs and preserves the raw content.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

try:  # Attempt optional imports of generated artifacts
    # These files are expected when running:
    #   java -jar grammer/antlr-4.13.1-complete.jar \
    #       -Dlanguage=Python3 -o parser grammer/EdgeFlow.g4
    # They may not be present in early development.
    from antlr4 import CommonTokenStream, FileStream  # type: ignore

    from .EdgeFlowLexer import EdgeFlowLexer  # type: ignore
    from .EdgeFlowParser import EdgeFlowParser  # type: ignore
    from .EdgeFlowVisitor import EdgeFlowVisitor  # type: ignore

    _ANTLR_AVAILABLE = True
except Exception:  # noqa: BLE001 - permissive import for optional dependency
    _ANTLR_AVAILABLE = False


logger = logging.getLogger(__name__)


def parse_ef(file_path: str) -> Dict[str, Any]:
    """Parse an EdgeFlow ``.ef`` file into a dictionary.

    Behavior:
    - If ANTLR-generated modules are available, parse with them and visit the
      tree to collect key/value assignments. The visitor here is minimal and
      should be replaced by Team A's rich visitor when available.
    - Otherwise, fallback to a line-based parser that supports ``key = value``.

    In both cases the result includes ``__source__`` and ``__raw__`` keys.

    Args:
        file_path: Path to the ``.ef`` configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration mapping.
    """

    # Always retain raw content for debugging regardless of parse route.
    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    if _ANTLR_AVAILABLE:
        try:

            class CollectVisitor(EdgeFlowVisitor):  # type: ignore[misc]
                def __init__(self) -> None:
                    self.data: Dict[str, Any] = {}

                # Generic visitor: try to collect assignments of form ID '=' value
                def visitChildren(self, node):  # type: ignore[override]
                    return super().visitChildren(node)

            # Tokenize and parse
            stream = FileStream(file_path, encoding="utf-8")
            lexer = EdgeFlowLexer(stream)  # type: ignore[call-arg]
            tokens = CommonTokenStream(lexer)
            parser = EdgeFlowParser(tokens)  # type: ignore[call-arg]
            tree = parser.start()  # type: ignore[attr-defined]

            # Visit tree. Without detailed grammar hooks here, we rely on
            # Team A's visitor to populate data. Keep a safe, empty default.
            visitor = CollectVisitor()
            visitor.visit(tree)
            result = dict(visitor.data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ANTLR parse failed, falling back to naive parse: %s", exc)
            result = {}
    else:
        result = {}

    # Naive fill if result is empty or partial; supports simple k = v lines.
    if not result:
        for line in raw_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                key, value = stripped.split("=", 1)
                k = key.strip()
                v = value.strip().strip('"')
                result[k] = v

    # Attach raw data for debugging/traceability.
    result.setdefault("__source__", file_path)
    result.setdefault("__raw__", "".join(raw_lines))

    logger.debug("Parsed EF config from %s: keys=%s", file_path, list(result.keys()))
    return result


# ---------------------------------------------------------------------------
# Day 2 parser API re-exports
# ---------------------------------------------------------------------------

try:
    # Load sibling top-level module parser.py using importlib to avoid name
    # collision with this package.
    import importlib.util
    import os
    from types import ModuleType

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    _mod_path = os.path.join(_root, "parser.py")
    if os.path.isfile(_mod_path):
        spec = importlib.util.spec_from_file_location("edgeflow_parser_core", _mod_path)
        if spec and spec.loader:  # type: ignore[truthy-bool]
            _core = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_core)  # type: ignore[arg-type]
            # Re-export selected symbols
            EdgeFlowParserError = getattr(_core, "EdgeFlowParserError")  # type: ignore[assignment]
            parse_edgeflow_file = getattr(_core, "parse_edgeflow_file")  # type: ignore[assignment]
            parse_edgeflow_string = getattr(_core, "parse_edgeflow_string")  # type: ignore[assignment]
            validate_config = getattr(_core, "validate_config")  # type: ignore[assignment]
            __all__ = [
                "parse_ef",
                "EdgeFlowParserError",
                "parse_edgeflow_file",
                "parse_edgeflow_string",
                "validate_config",
            ]
except Exception:  # noqa: BLE001 - do not fail package import if re-export fails
    pass
