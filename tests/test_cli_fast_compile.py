"""Tests for --fast-compile CLI path in edgeflowc."""

import argparse
from unittest.mock import Mock, patch

import edgeflowc


class TestFastCompilePath:
    """Test coverage for --fast-compile CLI functionality."""

    @patch("edgeflowc.parse_arguments")
    @patch("edgeflowc.fast_compile_config")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_fast_compile_success(
        self, mock_validate, mock_load, mock_fast_compile, mock_parse_args
    ):
        """Test successful fast compilation path."""
        # Setup mocks
        mock_load.return_value = {"model": "test.tflite", "target_device": "cpu"}

        mock_result = Mock()
        mock_result.success = True
        mock_result.compile_time_ms = 150.5
        mock_result.errors = []
        mock_result.warnings = ["Warning 1", "Warning 2"]
        mock_result.estimated_impact = {
            "estimated_size_reduction_percent": 25.3,
            "estimated_speed_improvement_factor": 1.5,
            "estimated_memory_reduction_percent": 30.0,
            "optimization_confidence": 0.85,
        }
        mock_fast_compile.return_value = mock_result

        # Create args with fast_compile
        mock_args = argparse.Namespace(
            config_path="test.ef",
            fast_compile=True,
            verbose=False,
            dry_run=False,
            check_only=False,
            docker=False,
            explain=False,
        )
        mock_parse_args.return_value = mock_args
        mock_validate.return_value = True

        # Run main
        result = edgeflowc.main()

        # Verify
        assert result == 0
        mock_load.assert_called_once_with("test.ef")
        mock_fast_compile.assert_called_once_with(mock_load.return_value)

    @patch("edgeflowc.parse_arguments")
    @patch("edgeflowc.fast_compile_config")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_fast_compile_failure(
        self, mock_validate, mock_load, mock_fast_compile, mock_parse_args
    ):
        """Test fast compilation failure path."""
        # Setup mocks
        mock_load.return_value = {"model": "test.tflite", "target_device": "cpu"}

        mock_result = Mock()
        mock_result.success = False
        mock_result.errors = ["Error 1: Invalid model", "Error 2: Unsupported ops"]
        mock_result.warnings = []
        mock_result.compile_time_ms = 50.0
        mock_result.estimated_impact = {}
        mock_fast_compile.return_value = mock_result

        # Create args with fast_compile
        mock_args = argparse.Namespace(
            config_path="test.ef",
            fast_compile=True,
            verbose=False,
            dry_run=False,
            check_only=False,
            docker=False,
            explain=False,
        )
        mock_parse_args.return_value = mock_args
        mock_validate.return_value = True

        # Run main
        result = edgeflowc.main()

        # Verify failure
        assert result == 1
        mock_load.assert_called_once_with("test.ef")
        mock_fast_compile.assert_called_once_with(mock_load.return_value)

    @patch("edgeflowc.parse_arguments")
    @patch("edgeflowc.fast_compile_config")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_fast_compile_no_warnings(
        self, mock_validate, mock_load, mock_fast_compile, mock_parse_args
    ):
        """Test fast compilation success without warnings."""
        # Setup mocks
        mock_load.return_value = {"model": "test.tflite", "target_device": "gpu"}

        mock_result = Mock()
        mock_result.success = True
        mock_result.compile_time_ms = 200.0
        mock_result.errors = []
        mock_result.warnings = []  # No warnings
        mock_result.estimated_impact = {
            "estimated_size_reduction_percent": 10.0,
            "estimated_speed_improvement_factor": 1.2,
            "estimated_memory_reduction_percent": 15.0,
            "optimization_confidence": 0.95,
        }
        mock_fast_compile.return_value = mock_result

        # Create args with fast_compile
        mock_args = argparse.Namespace(
            config_path="test.ef",
            fast_compile=True,
            verbose=False,
            dry_run=False,
            check_only=False,
            docker=False,
            explain=False,
        )
        mock_parse_args.return_value = mock_args
        mock_validate.return_value = True

        # Run main
        result = edgeflowc.main()

        # Verify success
        assert result == 0
        mock_load.assert_called_once()
        mock_fast_compile.assert_called_once()

    @patch("edgeflowc.parse_arguments")
    @patch("edgeflowc.fast_compile_config")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_fast_compile_with_verbose(
        self, mock_validate, mock_load, mock_fast_compile, mock_parse_args
    ):
        """Test fast compilation with verbose logging."""
        # Setup mocks
        mock_load.return_value = {"model": "test.tflite"}

        mock_result = Mock()
        mock_result.success = True
        mock_result.compile_time_ms = 100.0
        mock_result.errors = []
        mock_result.warnings = ["Performance warning"]
        mock_result.estimated_impact = {
            "estimated_size_reduction_percent": 20.0,
            "estimated_speed_improvement_factor": 1.3,
            "estimated_memory_reduction_percent": 25.0,
            "optimization_confidence": 0.9,
        }
        mock_fast_compile.return_value = mock_result

        # Create args with fast_compile and verbose
        mock_args = argparse.Namespace(
            config_path="test.ef",
            fast_compile=True,
            verbose=True,  # Enable verbose
            dry_run=False,
            check_only=False,
            docker=False,
            explain=False,
        )
        mock_parse_args.return_value = mock_args
        mock_validate.return_value = True

        # Run main
        with patch("logging.basicConfig") as mock_log_config:
            result = edgeflowc.main()
            # Verify verbose logging was configured
            mock_log_config.assert_called()
            call_kwargs = mock_log_config.call_args.kwargs
            assert call_kwargs.get("level") == 10  # DEBUG level

        # Verify success
        assert result == 0

    @patch("edgeflowc.parse_arguments")
    @patch("edgeflowc.fast_compile_config")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_fast_compile_with_parser_fallback(
        self, mock_validate, mock_load, mock_fast_compile, mock_parse_args
    ):
        """Test fast compilation with parser fallback when load_config fails."""
        # First call fails, triggering fallback
        mock_load.side_effect = [
            ImportError("Parser not available"),
            {"model": "test.tflite", "fallback": True},
        ]

        mock_result = Mock()
        mock_result.success = True
        mock_result.compile_time_ms = 80.0
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.estimated_impact = {
            "estimated_size_reduction_percent": 15.0,
            "estimated_speed_improvement_factor": 1.1,
            "estimated_memory_reduction_percent": 20.0,
            "optimization_confidence": 0.75,
        }
        mock_fast_compile.return_value = mock_result

        # Create args
        mock_args = argparse.Namespace(
            config_path="test.ef",
            fast_compile=True,
            verbose=False,
            dry_run=False,
            check_only=False,
            docker=False,
            explain=False,
        )
        mock_parse_args.return_value = mock_args
        mock_validate.return_value = True

        # Run main with fallback
        with patch("edgeflowc._load_config_fallback") as mock_fallback:
            mock_fallback.return_value = {"model": "test.tflite", "fallback": True}
            result = edgeflowc.main()

        # Verify success with fallback
        assert result == 0
        mock_fallback.assert_called_once_with("test.ef")
        mock_fast_compile.assert_called_once()
