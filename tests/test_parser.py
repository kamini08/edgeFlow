from edgeflow.parser import (
    EdgeFlowParserError,
    parse_edgeflow_file,
    parse_edgeflow_string,
    validate_config,
)
from pathlib import Path

import pytest


class TestEdgeFlowParser:
    """Test suite for EdgeFlow parser module."""

    def test_parse_valid_config(self):
        """Test parsing a valid configuration."""
        config_content = (
            "\n"
            'model_path = "model.tflite"\n'
            "quantize = int8\n"
            "batch_size = 32\n"
            "enable_pruning = true\n"
        )
        result = parse_edgeflow_string(config_content)
        assert result["model_path"] == "model.tflite"
        assert result["quantize"] == "int8"
        assert result["batch_size"] == 32
        assert result["enable_pruning"] is True

    def test_parse_all_value_types(self):
        """Test all supported value types: string, int, float, bool, identifier."""
        s = (
            's1 = "text"\n'
            "s2 = 'more'\n"
            "i = -123\n"
            "f1 = 3.14\n"
            "f2 = 1e-3\n"
            "b1 = true\n"
            "b2 = false\n"
            "id = latency\n"
        )
        out = parse_edgeflow_string(s)
        assert out["s1"] == "text"
        assert out["s2"] == "more"
        assert out["i"] == -123
        assert out["f1"] == 3.14
        assert abs(out["f2"] - 1e-3) < 1e-12
        assert out["b1"] is True
        assert out["b2"] is False
        assert out["id"] == "latency"

    def test_parse_with_comments(self):
        """Test that comments are properly ignored (including inline)."""
        s = (
            "# This is a comment\n"
            'model_path = "model.tflite"  # inline comment\n'
            "batch_size = 32 # another\n"
        )
        out = parse_edgeflow_string(s)
        assert out["model_path"] == "model.tflite"
        assert out["batch_size"] == 32

    def test_parse_empty_file(self):
        """Test parsing an empty configuration."""
        assert parse_edgeflow_string("") == {}

    def test_parse_syntax_error(self):
        """Test that syntax errors are properly reported."""
        with pytest.raises(EdgeFlowParserError) as exc_info:
            parse_edgeflow_string("invalid syntax ===")
        assert "syntax" in str(exc_info.value).lower()

    def test_parse_file_not_found(self):
        """Test file not found error."""
        with pytest.raises(FileNotFoundError):
            parse_edgeflow_file("nonexistent.ef")

    def test_validate_config_required_fields(self):
        """Test configuration validation for required fields and ranges."""
        ok, errs = validate_config({})
        assert not ok and any("model_path" in e for e in errs)

        ok, errs = validate_config({"model_path": "m.tflite", "batch_size": 0})
        assert not ok and any("batch_size" in e for e in errs)

        ok, errs = validate_config({"model_path": "m.tflite", "compression_ratio": 1.5})
        assert not ok and any("compression_ratio" in e for e in errs)

        ok, errs = validate_config({"model_path": "m.tflite", "enable_pruning": "yes"})
        assert not ok and any("enable_pruning" in e for e in errs)

        ok, errs = validate_config({"model_path": "m.tflite", "quantize": "int4"})
        assert not ok and any("quantize" in e for e in errs)

        ok, errs = validate_config({"model_path": "m.tflite", "optimize_for": "speed"})
        assert not ok and any("optimize_for" in e for e in errs)

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ('key = "value"', {"key": "value"}),
            ("key = 123", {"key": 123}),
            ("key = 3.14", {"key": 3.14}),
            ("key = true", {"key": True}),
            ("key = false", {"key": False}),
        ],
    )
    def test_parse_parametrized(self, input_str, expected):
        """Parametrized tests for different input types."""
        result = parse_edgeflow_string(input_str)
        assert result == expected


class TestParserIntegration:
    """Basic integration checks without starting the HTTP server."""

    def test_parser_with_cli(self, tmp_path: Path, monkeypatch):
        """Test parser integration via CLI load path (no subprocess)."""
        p = tmp_path / "c.ef"
        p.write_text('model = "x.tflite"\nquantize = int8\n', encoding="utf-8")

        # Use CLI's load_config directly to avoid dependency on optimizer
        import importlib

        from edgeflow.compiler import edgeflowc

        importlib.reload(edgeflowc)
        cfg = edgeflowc.load_config(str(p))
        assert cfg["model"] == "x.tflite"
        assert cfg["quantize"] == "int8"
