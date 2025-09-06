from __future__ import annotations

from backend.api.services.parser_service import ParserService


def test_parser_service_success():
    content = 'model_path = "m.tflite"\nquantize = int8\n'
    ok, cfg, err = ParserService.parse_config_content(content)
    assert ok is True
    assert cfg["model_path"] == "m.tflite"
    assert err == ""


def test_parser_service_validation_error():
    content = "quantize = int8\n"  # missing model_path
    ok, cfg, err = ParserService.parse_config_content(content)
    assert ok is False
    assert cfg == {}
    assert "model_path" in err


def test_parser_service_syntax_error():
    bad = "invalid syntax ==="
    ok, cfg, err = ParserService.parse_config_content(bad)
    assert ok is False and cfg == {}
    assert "syntax" in err.lower()


def test_parser_service_unexpected_error(monkeypatch):
    # Force a generic exception path by monkeypatching module-level function
    import backend.api.services.parser_service as ps

    def boom(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(ps, "parse_edgeflow_string", boom)
    ok, cfg, err = ParserService.parse_config_content("model_path='x'")
    assert ok is False and cfg == {}
    assert "Unexpected error" in err
