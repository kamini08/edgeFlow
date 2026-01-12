import json
import tempfile
from pathlib import Path

import pytest
from edgeflow.reporting.reporter import (
    calculate_improvements,
    generate_json_report,
    generate_recommendations,
    generate_report,
)


class TestReporter:
    """Test suite for the reporter module."""

    @pytest.fixture
    def sample_stats(self):
        """Provide sample statistics for testing."""
        return {
            "unoptimized": {
                "size_mb": 10.5,
                "latency_ms": 25.0,
                "model_path": "models/original.tflite",
            },
            "optimized": {
                "size_mb": 2.8,
                "latency_ms": 8.5,
                "model_path": "models/optimized.tflite",
            },
        }

    @pytest.fixture
    def sample_config(self):
        """Provide sample configuration."""
        return {
            "model_path": "models/original.tflite",
            "quantize": "int8",
            "target_device": "raspberry_pi",
            "optimize_for": "latency",
        }

    def test_generate_report_creates_file(self, sample_stats, sample_config):
        """Test that report generation creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.md"

            result = generate_report(
                sample_stats["unoptimized"],
                sample_stats["optimized"],
                sample_config,
                str(output_path),
            )

            assert output_path.exists()
            assert output_path.stat().st_size > 0
            assert result == str(output_path)

    def test_report_contains_required_sections(self, sample_stats, sample_config):
        """Test that report contains all required sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.md"

            generate_report(
                sample_stats["unoptimized"],
                sample_stats["optimized"],
                sample_config,
                str(output_path),
            )

            content = output_path.read_text()

            # Check for required sections
            assert "# EdgeFlow Optimization Report" in content
            assert "## Executive Summary" in content
            assert "## Performance Metrics" in content
            assert "## Configuration Used" in content
            assert "## Optimization Details" in content

            # Check for metrics
            assert "10.5" in content  # Original size
            assert "2.8" in content  # Optimized size
            assert "25.0" in content  # Original latency
            assert "8.5" in content  # Optimized latency

    def test_calculate_improvements(self, sample_stats):
        """Test improvement calculation accuracy."""
        improvements = calculate_improvements(
            sample_stats["unoptimized"], sample_stats["optimized"]
        )

        assert improvements["size_reduction"] == pytest.approx(73.33, rel=0.01)
        assert improvements["speedup"] == pytest.approx(2.94, rel=0.01)
        assert improvements["latency_reduction"] == pytest.approx(66.0, rel=0.01)

    def test_missing_required_fields(self):
        """Test that missing fields raise appropriate errors."""
        incomplete_stats = {"size_mb": 10.0}  # Missing latency_ms

        with pytest.raises(ValueError, match="Missing required field"):
            generate_report(incomplete_stats, incomplete_stats)

    def test_json_report_generation(self, sample_stats, sample_config):
        """Test JSON report generation for API consumption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            generate_json_report(
                sample_stats["unoptimized"],
                sample_stats["optimized"],
                sample_config,
                str(output_path),
            )

            assert output_path.exists()

            # Validate JSON structure
            with open(output_path) as f:
                data = json.load(f)
                assert "original_stats" in data
                assert "optimized_stats" in data
                assert "improvements" in data
                assert "timestamp" in data

    @pytest.mark.parametrize(
        "size_reduction,expected_recommendation",
        [
            (75, "Excellent size reduction"),
            (50, "Good size reduction"),
            (25, "Moderate size reduction"),
        ],
    )
    def test_contextual_recommendations(self, size_reduction, expected_recommendation):
        """Test that recommendations are contextual."""
        improvements = {
            "size_reduction": float(size_reduction),
            "speedup": 1.0,
            "latency_reduction": 0.0,
            "throughput_increase": 0.0,
            "memory_saved": 0.0,
        }
        rec = generate_recommendations(improvements)
        assert expected_recommendation in rec
