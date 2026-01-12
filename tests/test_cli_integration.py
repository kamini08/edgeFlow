import os
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestEndToEndIntegration:
    """End-to-end integration tests (CLI and optional API)."""

    def test_cli_parser_integration(self):
        import os, sys; print(f"DEBUG: PYTHONPATH: {os.environ.get("PYTHONPATH")}"); print(f"DEBUG: sys.path: {sys.path}")
        import sys; print(f"DEBUG: sys.executable: {sys.executable}")
        """Test CLI with parser end-to-end using --dry-run."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ef", delete=False) as f:
            content = """
model = "test.tflite"
quantize = int8
"""
            f.write(content)
            config_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "edgeflow.compiler.edgeflowc", config_path,  "--check-only"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert "model" in result.stdout
        finally:
            Path(config_path).unlink(missing_ok=True)

    @pytest.mark.skipif(
        os.environ.get("RUN_API_INTEGRATION", "0") != "1",
        reason="HTTP API not started; set RUN_API_INTEGRATION=1 to enable",
    )
    def test_api_parser_integration(self):
        """Test API with parser end-to-end (requires running API)."""
        requests = pytest.importorskip("requests")
        config_content = """
model = "test.tflite"
quantize = int8
"""

        response = requests.post(
            "http://localhost:8000/api/compile",
            json={"config_file": config_content, "filename": "test.ef"},
            timeout=5,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["parsed_config"]["model"] == "test.tflite"
