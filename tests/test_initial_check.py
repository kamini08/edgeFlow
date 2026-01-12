import json
import os
from pathlib import Path

import pytest
from edgeflow.config.device_specs import DeviceSpecManager
from edgeflow.analysis.initial_check import CompatibilityReport, InitialChecker, ModelProfile


class TestInitialCheck:
    """Test suite for initial check module."""

    @pytest.fixture
    def sample_model(self, tmp_path: Path) -> str:
        """Create a sample lightweight model file for testing."""
        p = tmp_path / "tiny_int8_model.tflite"
        # ~100KB file (heuristic sufficient for profiling by size)
        p.write_bytes(os.urandom(100 * 1024))
        return str(p)

    def test_profile_model(self, sample_model):
        """Test model profiling functionality."""
        checker = InitialChecker()
        profile = checker.profile_model(sample_model)

        assert isinstance(profile, ModelProfile)
        assert profile.file_size_mb > 0
        assert profile.num_parameters > 0
        assert profile.estimated_ram_mb >= profile.file_size_mb

    def test_compatibility_check_pass(self, sample_model):
        """Test successful compatibility check against RPi4 defaults."""
        checker = InitialChecker()
        cfg = {
            "target_device": "raspberry_pi_4",
            "quantize": "int8",
            "model_path": sample_model,
        }
        report = checker.check_compatibility(sample_model, cfg["target_device"], cfg)
        assert isinstance(report, CompatibilityReport)
        assert report.estimated_fit_score > 0
        assert report.compatible is True

    def test_compatibility_check_fail(self, tmp_path: Path):
        """Test failed compatibility check when model grossly exceeds memory limits."""
        # Create a large dummy model (5MB) which exceeds ESP32 limits
        big = tmp_path / "huge_model.tflite"
        big.write_bytes(b"0" * (5 * 1024 * 1024))
        checker = InitialChecker()
        cfg = {"target_device": "esp32", "quantize": "int8", "model_path": str(big)}
        report = checker.check_compatibility(str(big), cfg["target_device"], cfg)
        assert report.compatible is False
        assert any("exceeds memory limit" in msg for msg in report.issues)

    def test_device_spec_loading(self, tmp_path: Path):
        """Test loading custom device specifications from CSV and JSON."""
        # CSV
        csv_path = tmp_path / "devices.csv"
        csv_path.write_text(
            "name,ram_mb,storage_mb,max_model_size_mb,cpu_cores\ncustom_device,1024,4096,100,2\n",
            encoding="utf-8",
        )
        mgr_csv = DeviceSpecManager(str(csv_path))
        assert "custom_device" in mgr_csv.devices
        # JSON
        json_path = tmp_path / "devices.json"
        json_path.write_text(
            json.dumps(
                {
                    "devices": [
                        {
                            "name": "custom_json",
                            "ram_mb": 2048,
                            "max_model_size_mb": 150,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        mgr_json = DeviceSpecManager(str(json_path))
        assert "custom_json" in mgr_json.devices

    def test_fit_score_calculation(self, sample_model):
        """Test fit score calculation logic yields reasonable range."""
        checker = InitialChecker()
        prof = checker.profile_model(sample_model)
        spec = checker.spec_manager.get_device_spec("raspberry_pi_4")
        score = checker._calculate_fit_score(prof, spec, spec.ram_mb)
        assert 0 <= score <= 100

    @pytest.mark.integration
    def test_cli_integration(self, tmp_path: Path, monkeypatch):
        """Test CLI integration with --check-only flag."""
        # Write a minimal config file
        model = tmp_path / "m.tflite"
        model.write_bytes(os.urandom(50 * 1024))
        ef = tmp_path / "test.ef"
        ef.write_text(
            f"model = '{model}'\n target_device = 'raspberry_pi'\n",
            encoding="utf-8",
        )

        # Import CLI and run main() with args
        from edgeflow.compiler import edgeflowc as cli

        monkeypatch.setenv("PYTHONWARNINGS", "ignore")
        monkeypatch.setattr(
            "sys.argv",
            ["edgeflowc.py", str(ef), "--check-only"],
        )  # type: ignore[attr-defined]
        code = cli.main()
        assert code == 0

    @pytest.mark.integration
    def test_api_integration(self, sample_model):
        """Test API endpoint for compatibility checking.

        Skip when FastAPI is not available in the environment.
        """
        try:
            from backend.app import app  # type: ignore
            from fastapi.testclient import TestClient  # type: ignore
        except Exception:
            pytest.skip("fastapi not installed; skipping API integration test")
            return

        client = TestClient(app)
        payload = {
            "model_path": sample_model,
            "config": {"target_device": "raspberry_pi_4"},
        }
        r = client.post("/api/check", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert set(["compatible", "requires_optimization", "fit_score"]).issubset(
            data.keys()
        )
