from __future__ import annotations

import pytest

from backend import app as appmod


def test__parse_config_content_success_path(monkeypatch):
    # Ensure normal success path returns parsed cfg from ParserService
    content = 'model_path = "x.tflite"\n'
    cfg = appmod._parse_config_content("x.ef", content)
    assert cfg["model_path"] == "x.tflite"


def test__parse_config_content_fallback_to_parse_ef(monkeypatch):
    # Force ParserService to signal failure to trigger fallback to parse_ef
    class Dummy:
        @staticmethod
        def parse_config_content(_):
            return False, {}, "boom"

    monkeypatch.setattr(appmod, "ParserService", Dummy)
    content = 'model_path = "x.tflite"\n'
    cfg = appmod._parse_config_content("x.ef", content)
    assert cfg.get("__source__") or cfg.get("model_path")


def test__parse_config_content_invalid_extension():
    with pytest.raises(Exception):
        appmod._parse_config_content("x.txt", "x=1")


def test__b64_size_mb_valid_small():
    # Valid base64 for empty content should be 0 MB
    assert appmod._b64_size_mb("") == 0.0
