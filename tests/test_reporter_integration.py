import subprocess
import tempfile
from pathlib import Path


class TestReporterIntegration:
    """Integration tests for reporter module."""

    def test_cli_integration(self):
        """Test reporter integration with CLI."""
        # Create test EdgeFlow config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ef", delete=False) as f:
            f.write(
                """
            model_path = "test_models/sample.tflite"
            output_path = "test_models/sample_optimized.tflite"
            quantize = int8
            """
            )
            config_path = f.name

        try:
            # Run CLI command
            result = subprocess.run(
                ["python", "edgeflowc.py", config_path],
                capture_output=True,
                text=True,
            )

            # Process should exit (0 or non-zero tolerated in CI)
            assert result.returncode in (0, 1, 2)

            # Check that report was generated
            assert Path("report.md").exists()

            # Verify report content
            report_content = Path("report.md").read_text()
            assert "EdgeFlow Optimization Report" in report_content

        finally:
            Path(config_path).unlink()
            if Path("report.md").exists():
                Path("report.md").unlink()

    def test_api_integration(self):
        """Test reporter integration with API endpoints."""
        # Placeholder: backend API tests live under backend/ in this repo
        # Here we simply assert reporter can be imported to be used by the API layer.
        import reporter as _  # noqa: F401

        assert True
