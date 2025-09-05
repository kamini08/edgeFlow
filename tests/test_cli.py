import sys
from types import ModuleType
from typing import Dict

import pytest

import edgeflowc


def _set_argv(args):
    sys.argv = ["edgeflowc.py", *args]


def test_parse_arguments_with_config_path(monkeypatch):
    _set_argv(["model.ef"])
    ns = edgeflowc.parse_arguments()
    assert ns.config_path == "model.ef"
    assert ns.verbose is False


def test_parse_arguments_verbose(monkeypatch):
    _set_argv(["model.ef", "--verbose"])
    ns = edgeflowc.parse_arguments()
    assert ns.verbose is True


def test_parse_arguments_help(monkeypatch):
    _set_argv(["--help"])
    with pytest.raises(SystemExit) as exc:
        edgeflowc.parse_arguments()
    assert exc.value.code == 0


def test_parse_arguments_version(monkeypatch, capsys):
    _set_argv(["--version"])
    with pytest.raises(SystemExit) as exc:
        edgeflowc.parse_arguments()
    assert exc.value.code == 0


def test_validate_file_path_nonexistent(tmp_path):
    assert edgeflowc.validate_file_path(str(tmp_path / "missing.ef")) is False


def test_validate_file_path_wrong_extension(tmp_path):
    p = tmp_path / "config.txt"
    p.write_text("model_path=...", encoding="utf-8")
    assert edgeflowc.validate_file_path(str(p)) is False


def test_validate_file_path_directory(tmp_path):
    assert edgeflowc.validate_file_path(str(tmp_path)) is False


def test_validate_file_path_uppercase_extension(tmp_path):
    p = tmp_path / "CONFIG.EF"
    p.write_text("quantize=int8", encoding="utf-8")
    assert edgeflowc.validate_file_path(str(p)) is True


def test_load_config_fallback_reads_file(tmp_path):
    p = tmp_path / "model.ef"
    content = 'model_path="m.tflite"\n'
    p.write_text(content, encoding="utf-8")
    cfg = edgeflowc.load_config(str(p))
    assert cfg["__source__"].endswith("model.ef")
    assert cfg["__raw__"] == content


def test_load_config_uses_parser_if_available(tmp_path, monkeypatch):
    # Inject a fake parser module
    fake = ModuleType("parser")

    def parse_ef(path: str) -> Dict[str, str]:
        return {"parsed": path}

    fake.parse_ef = parse_ef  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "parser", fake)

    p = tmp_path / "conf.ef"
    p.write_text("x=1", encoding="utf-8")
    cfg = edgeflowc.load_config(str(p))
    assert cfg == {"parsed": str(p)}


def test_optimize_model_uses_optimizer_if_available(monkeypatch):
    called = {"ok": False}
    fake = ModuleType("optimizer")

    def optimize(cfg):  # pragma: no cover (covered via call below)
        called["ok"] = True

    fake.optimize = optimize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "optimizer", fake)

    edgeflowc.optimize_model({"k": 1})
    assert called["ok"] is True


def test_optimize_model_handles_exception(monkeypatch, caplog):
    # Provide an optimizer that raises an exception to hit the except path
    fake = ModuleType("optimizer")

    def optimize(cfg):  # pragma: no cover - asserted via logging
        raise RuntimeError("boom")

    fake.optimize = optimize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "optimizer", fake)

    caplog.set_level("INFO")
    edgeflowc.optimize_model({})
    assert any("Optimizer not available" in r.message for r in caplog.records)


def test_main_no_args_returns_2(monkeypatch):
    _set_argv([])
    assert edgeflowc.main() == 2


def test_main_nonexistent_file_returns_1(monkeypatch):
    _set_argv(["missing.ef"])
    assert edgeflowc.main() == 1


def test_main_invalid_extension_returns_1(tmp_path, monkeypatch):
    p = tmp_path / "invalid.txt"
    p.write_text("x=1", encoding="utf-8")
    _set_argv([str(p)])
    assert edgeflowc.main() == 1


def test_main_success_calls_optimize(tmp_path, monkeypatch):
    p = tmp_path / "ok.ef"
    p.write_text("x=1", encoding="utf-8")
    called = {"n": 0}

    def fake_opt(config):
        called["n"] += 1

    monkeypatch.setattr(edgeflowc, "optimize_model", fake_opt)
    _set_argv([str(p)])
    code = edgeflowc.main()
    assert code == 0
    assert called["n"] == 1


def test_main_verbose_emits_debug_log(tmp_path, monkeypatch, caplog):
    p = tmp_path / "ok.ef"
    p.write_text("x=1", encoding="utf-8")
    monkeypatch.setattr(edgeflowc, "optimize_model", lambda cfg: None)

    caplog.set_level("DEBUG")
    _set_argv([str(p), "--verbose"])
    code = edgeflowc.main()
    assert code == 0
    # confirm debug log emitted by load step
    assert any("Loaded config" in r.message for r in caplog.records)


def test_main_help_returns_0(monkeypatch):
    # Verify SystemExit handling path in main
    _set_argv(["--help"]) 
    rc = edgeflowc.main()
    assert rc == 0


def test_main_version_returns_0():
    _set_argv(["--version"]) 
    rc = edgeflowc.main()
    assert rc == 0


def test_main_handles_unexpected_exception(monkeypatch):
    # Force an unexpected exception inside main
    def boom():
        raise RuntimeError("unexpected")

    monkeypatch.setattr(edgeflowc, "parse_arguments", boom)
    rc = edgeflowc.main()
    assert rc == 1


def test_validate_file_path_empty_string():
    assert edgeflowc.validate_file_path("") is False


def test_validate_file_path_normpath_exception(monkeypatch):
    # Monkeypatch os.path.normpath to raise
    import os as _os

    def bad_norm(p):
        raise ValueError("bad")

    monkeypatch.setattr(_os.path, "normpath", bad_norm)
    assert edgeflowc.validate_file_path("foo.ef") is False
