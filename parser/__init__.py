"""EdgeFlow DSL parser interface.

Exposes :func:`parse_ef` which attempts to use ANTLR-generated modules
(`EdgeFlowLexer`, `EdgeFlowParser`, `EdgeFlowVisitor`) if present under this
package. If they are not available, it falls back to a simple line-based parse
that supports ``key = value`` pairs and preserves the raw content.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

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
    has_antlr = True
except Exception:  # noqa: BLE001 - permissive import for optional dependency
    _ANTLR_AVAILABLE = False
    has_antlr = False


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

                def visitModelStmt(self, ctx):  # type: ignore[misc]
                    string_token = ctx.STRING()
                    if string_token:
                        # Remove quotes from string
                        value = string_token.getText()[1:-1]
                        self.data["model"] = value
                    return self.visitChildren(ctx)

                def visitQuantizeStmt(self, ctx):  # type: ignore[misc]
                    quant_type = ctx.quantType()
                    if quant_type:
                        self.data["quantize"] = quant_type.getText()
                    return self.visitChildren(ctx)

                def visitTargetDeviceStmt(self, ctx):  # type: ignore[misc]
                    identifier = ctx.IDENTIFIER()
                    if identifier:
                        self.data["target_device"] = identifier.getText()
                    return self.visitChildren(ctx)

                def visitDeployPathStmt(self, ctx):  # type: ignore[misc]
                    string_token = ctx.STRING()
                    if string_token:
                        value = string_token.getText()[1:-1]
                        self.data["deploy_path"] = value
                    return self.visitChildren(ctx)

                def visitInputStreamStmt(self, ctx):  # type: ignore[misc]
                    identifier = ctx.IDENTIFIER()
                    if identifier:
                        self.data["input_stream"] = identifier.getText()
                    return self.visitChildren(ctx)

                def visitBufferSizeStmt(self, ctx):  # type: ignore[misc]
                    integer = ctx.INTEGER()
                    if integer:
                        self.data["buffer_size"] = int(integer.getText())
                    return self.visitChildren(ctx)

                def visitOptimizeForStmt(self, ctx):  # type: ignore[misc]
                    identifier = ctx.IDENTIFIER()
                    if identifier:
                        self.data["optimize_for"] = identifier.getText()
                    return self.visitChildren(ctx)

                def visitMemoryLimitStmt(self, ctx):  # type: ignore[misc]
                    integer = ctx.INTEGER()
                    if integer:
                        self.data["memory_limit"] = int(integer.getText())
                    return self.visitChildren(ctx)

                def visitFusionStmt(self, ctx):  # type: ignore[misc]
                    bool_token = ctx.BOOL()
                    if bool_token:
                        self.data["enable_fusion"] = bool_token.getText() == "true"
                    return self.visitChildren(ctx)

            # Tokenize and parse
            stream = FileStream(file_path, encoding="utf-8")
            lexer = EdgeFlowLexer(stream)  # type: ignore[call-arg]
            tokens = CommonTokenStream(lexer)
            parser = EdgeFlowParser(tokens)  # type: ignore[call-arg]
            tree = parser.program()  # type: ignore[attr-defined]
            
            # Visit the tree to collect data
            visitor = CollectVisitor()
            visitor.visit(tree)
            result = visitor.data

        except Exception as exc:  # noqa: BLE001
            logger.warning("ANTLR parse failed, falling back to naive parse: %s", exc)
            result = {}
    else:
        result = {}

    # Strict fill if result is empty or partial; supports simple k = v lines.
    if not result:
        result = _strict_kv_from_lines(raw_lines)

    # Attach raw data for debugging/traceability.
    result.setdefault("__source__", file_path)
    result.setdefault("__raw__", "".join(raw_lines))

    logger.debug("Parsed EF config from %s: keys=%s", file_path, list(result.keys()))
    return result


# ---------------------------------------------------------------------------
# Day 2 parser API re-exports
# ---------------------------------------------------------------------------

# Static type stubs so mypy sees these attributes on the package
if TYPE_CHECKING:

    class EdgeFlowParserError(Exception):
        pass

    def parse_edgeflow_string(content: str) -> Dict[str, Any]:
        pass

    def parse_edgeflow_file(file_path: str) -> Dict[str, Any]:
        pass

    def validate_config(cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
        pass


def _ensure_day2_exports() -> None:
    """Best-effort re-export of Day 2 parser API from top-level parser.py.

    Guarantees that ``EdgeFlowParserError``, ``parse_edgeflow_string``,
    ``parse_edgeflow_file`` and ``validate_config`` are available from this
    package even if importlib tricks fail in some environments.
    """

    import importlib.util
    import os

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    mod_path = os.path.join(root, "parser.py")
    if os.path.isfile(mod_path):
        try:
            spec = importlib.util.spec_from_file_location(
                "edgeflow_parser_core", mod_path
            )
            if spec and spec.loader:  # type: ignore[truthy-bool]
                core = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(core)  # type: ignore[arg-type]
                globals()["EdgeFlowParserError"] = getattr(core, "EdgeFlowParserError")
                globals()["parse_edgeflow_file"] = getattr(core, "parse_edgeflow_file")
                globals()["parse_edgeflow_string"] = getattr(
                    core, "parse_edgeflow_string"
                )
                globals()["validate_config"] = getattr(core, "validate_config")
        except Exception:
            # Fall through to minimal safe fallbacks below
            pass

    # If still missing, provide minimal fallbacks that use Day 1 API
    if "EdgeFlowParserError" not in globals():

        class _EdgeFlowParserError(Exception):
            """Fallback parser error type."""

        globals()["EdgeFlowParserError"] = _EdgeFlowParserError

    if "parse_edgeflow_string" not in globals():

        def parse_edgeflow_string(  # type: ignore[name-defined]
            content: str,
        ) -> Dict[str, Any]:
            # Write content to a temp file and reuse parse_ef
            import tempfile

            with tempfile.NamedTemporaryFile("w", suffix=".ef", delete=False) as tf:
                tf.write(content)
                path = tf.name
            try:
                return parse_ef(path)
            finally:
                try:
                    os.unlink(path)
                except Exception:
                    pass

        globals()["parse_edgeflow_string"] = parse_edgeflow_string

    if "parse_edgeflow_file" not in globals():

        def parse_edgeflow_file(  # type: ignore[name-defined]
            file_path: str,
        ) -> Dict[str, Any]:
            return parse_ef(file_path)

        globals()["parse_edgeflow_file"] = parse_edgeflow_file

    if "validate_config" not in globals():

        def _validate_config(  # type: ignore[name-defined]
            cfg: Dict[str, Any],
        ) -> Tuple[bool, List[str]]:
            # Minimal validation: ensure a string model_path exists
            ok = isinstance(cfg.get("model_path"), str) and bool(
                cfg["model_path"].strip()
            )
            errs: List[str] = (
                []
                if ok
                else [
                    "'model_path' is required and must be a non-empty string",
                ]
            )
            return ok, errs

        globals()["validate_config"] = _validate_config

    globals()["__all__"] = [
        "parse_ef",
        "EdgeFlowParserError",
        "parse_edgeflow_file",
        "parse_edgeflow_string",
        "validate_config",
    ]


# Attempt to re-export from top-level parser.py and guarantee API presence
_ensure_day2_exports()


def _strict_kv_from_lines(raw_lines: List[str]) -> Dict[str, Any]:
    """Parse lines into key/value pairs with minimal validation.

    - Skips blank and comment lines.
    - Requires exactly one '=' per line (outside quotes).
    - Requires non-empty key and value.
    - Strips surrounding quotes on values.
    - Raises EdgeFlowParserError on any syntax issue.
    """

    result: Dict[str, Any] = {}
    errors: List[str] = []

    for lineno, raw in enumerate(raw_lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        in_s = False
        in_d = False
        eq_positions: List[int] = []
        for idx, ch in enumerate(raw):
            if ch == "'" and not in_d:
                in_s = not in_s
            elif ch == '"' and not in_s:
                in_d = not in_d
            elif ch == "=" and not in_s and not in_d:
                eq_positions.append(idx)

        if len(eq_positions) != 1:
            errors.append(
                f"Line {lineno}: syntax error - expected single '=' in assignment"
            )
            continue

        eq = eq_positions[0]
        key = raw[:eq].strip()
        val = raw[eq + 1 :].strip()

        if not key:
            errors.append(f"Line {lineno}: syntax error - missing key before '='")
            continue
        if not val:
            errors.append(f"Line {lineno}: syntax error - missing value after '='")
            continue

        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]

        result[key] = val

    if errors:
        err_type = globals().get("EdgeFlowParserError", Exception)
        raise err_type("; ".join(errors))  # type: ignore[misc]

    return result
