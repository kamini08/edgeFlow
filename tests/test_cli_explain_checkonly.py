"""Tests for --explain and --check-only CLI paths in edgeflowc."""

import argparse
import json
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

import edgeflowc


class TestExplainPath:
    """Test coverage for --explain CLI functionality."""

    @patch("edgeflowc.parse_arguments")
    @patch("edgeflowc.generate_explainability_report")
    @patch("edgeflowc.optimize_model")
    @patch("edgeflowc.CodeGenerator")
    @patch("edgeflowc.apply_ir_transformations")
    @patch("edgeflowc.IRBuilder")
    @patch("edgeflowc.create_program_from_dict")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_explain_after_optimization(
        self,
        mock_validate,
        mock_load,
        mock_create_program,
        mock_ir_builder,
        mock_apply_ir,
        mock_codegen,
        mock_optimize,
        mock_explain_report,
        mock_parse_args,
    ):
        """Test --explain flag triggers explanation after optimization."""
        # Setup mocks
        mock_validate.return_value = True
        mock_load.return_value = {"model": "test.tflite", "target_device": "cpu"}
        mock_program = Mock()
        mock_program.statements = [Mock(), Mock()]
        mock_create_program.return_value = mock_program

        mock_ir_graph = Mock()
        mock_ir_graph.nodes = [1, 2, 3]
        mock_ir_graph.edges = [1, 2]
        mock_ir_builder.return_value.build_from_config.return_value = mock_ir_graph

        mock_apply_ir.return_value = {"passes_applied": 5}

        mock_generator = Mock()
        mock_generator.generate_python_inference.return_value = "# Python code"
        mock_generator.generate_cpp_inference.return_value = "// C++ code"
        mock_codegen.return_value = mock_generator

        mock_optimize.return_value = (True, "Optimization successful", {"metrics": {}})
        mock_explain_report.return_value = "Detailed optimization explanation"

        # Create args with explain
        mock_args = argparse.Namespace(
            config_path="test.ef",
            explain=True,
            verbose=False,
            dry_run=False,
            check_only=False,
            docker=False,
            fast_compile=False,
        )
        mock_parse_args.return_value = mock_args

        # Run main
        with patch("os.makedirs"):
            with patch("builtins.open", create=True) as mock_open:
                result = edgeflowc.main()

        # Verify
        assert result == 0
        mock_load.assert_called_once()
        mock_optimize.assert_called_once()
        mock_explain_report.assert_called_once()

    @patch("edgeflowc.parse_arguments")
    @patch("edgeflowc.optimize_model")
    @patch("edgeflowc.CodeGenerator")
    @patch("edgeflowc.apply_ir_transformations")
    @patch("edgeflowc.IRBuilder")
    @patch("edgeflowc.create_program_from_dict")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_no_explain_without_flag(
        self,
        mock_validate,
        mock_load,
        mock_create_program,
        mock_ir_builder,
        mock_apply_ir,
        mock_codegen,
        mock_optimize,
        mock_parse_args,
    ):
        """Test that explanation is not called without --explain flag."""
        # Setup mocks
        mock_validate.return_value = True
        mock_load.return_value = {"model": "test.tflite"}
        mock_program = Mock()
        mock_program.statements = []
        mock_create_program.return_value = mock_program

        mock_ir_graph = Mock()
        mock_ir_graph.nodes = []
        mock_ir_graph.edges = []
        mock_ir_builder.return_value.build_from_config.return_value = mock_ir_graph

        mock_apply_ir.return_value = {}
        mock_codegen.return_value.generate_python_inference.return_value = ""
        mock_codegen.return_value.generate_cpp_inference.return_value = ""
        mock_optimize.return_value = (True, "Success", {})

        # Create args WITHOUT explain
        mock_args = argparse.Namespace(
            config_path="test.ef",
            explain=False,  # No explanation
            verbose=False,
            dry_run=False,
            check_only=False,
            docker=False,
            fast_compile=False,
        )
        mock_parse_args.return_value = mock_args

        # Run main with patched explain to ensure it's not called
        with patch("edgeflowc.generate_explainability_report") as mock_explain:
            with patch("os.makedirs"):
                with patch("builtins.open", create=True):
                    result = edgeflowc.main()
            mock_explain.assert_not_called()

        assert result == 0


class TestCheckOnlyPath:
    """Test coverage for --check-only CLI functionality."""

    @patch("edgeflowc.parse_arguments")
    @patch("initial_check.perform_initial_check")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_check_only_with_model(self, mock_validate, mock_load, mock_initial_check, mock_parse_args):
        """Test --check-only with model specified."""
        # Setup mocks
        mock_validate.return_value = True
        mock_load.return_value = {
            "model": "test.tflite",
            "target_device": "edge_tpu",
            "model_path": "test.tflite",
        }

        mock_report = Mock()
        mock_report.estimated_fit_score = 85.5
        mock_report.issues = ["Issue 1: Large model size", "Issue 2: Unsupported ops"]
        mock_report.recommendations = [
            "Recommendation 1: Apply quantization",
            "Recommendation 2: Use model pruning",
        ]
        mock_initial_check.return_value = (True, mock_report)

        # Create args with check_only
        mock_args = argparse.Namespace(
            config_path="test.ef",
            check_only=True,
            verbose=False,
            dry_run=False,
            docker=False,
            fast_compile=False,
            explain=False,
            device_spec_file=None,
        )
        mock_parse_args.return_value = mock_args

        # Run main
        result = edgeflowc.main()

        # Verify
        assert result == 0
        mock_load.assert_called_once()
        mock_initial_check.assert_called_once_with("test.tflite", mock_load.return_value, None)

    @patch("edgeflowc.parse_arguments")
    @patch("initial_check.perform_initial_check")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_check_only_without_model(self, mock_validate, mock_load, mock_initial_check, mock_parse_args):
        """Test --check-only when no model is specified (skip check)."""
        # Setup mocks - no model in config
        mock_validate.return_value = True
        mock_load.return_value = {
            "target_device": "cpu",
            # No model or model_path
        }

        # Create args with check_only
        mock_args = argparse.Namespace(
            config_path="test.ef",
            check_only=True,
            verbose=False,
            dry_run=False,
            docker=False,
            fast_compile=False,
            explain=False,
        )
        mock_parse_args.return_value = mock_args

        # Run main - should skip check and not call perform_initial_check
        with patch("logging.warning") as mock_warning:
            result = edgeflowc.main()
            # Verify warning was logged about no model
            mock_warning.assert_called()

        # Should not attempt initial check
        mock_initial_check.assert_not_called()
        assert result == 0

    @patch("edgeflowc.parse_arguments")
    @patch("initial_check.perform_initial_check")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_check_only_no_issues(self, mock_validate, mock_load, mock_initial_check, mock_parse_args):
        """Test --check-only with no issues found."""
        # Setup mocks
        mock_validate.return_value = True
        mock_load.return_value = {
            "model_path": "optimized.tflite",
            "target_device": "cpu",
        }

        mock_report = Mock()
        mock_report.estimated_fit_score = 98.0
        mock_report.issues = []  # No issues
        mock_report.recommendations = []  # No recommendations
        mock_initial_check.return_value = (False, mock_report)  # Model fits

        # Create args
        mock_args = argparse.Namespace(
            config_path="test.ef",
            check_only=True,
            verbose=False,
            dry_run=False,
            docker=False,
            fast_compile=False,
            explain=False,
            device_spec_file=None,
        )
        mock_parse_args.return_value = mock_args

        # Run main
        result = edgeflowc.main()

        # Verify clean exit
        assert result == 0
        mock_initial_check.assert_called_once()

    @patch("edgeflowc.parse_arguments")
    @patch("initial_check.perform_initial_check")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_initial_check_exception_handling(self, mock_validate, mock_load, mock_initial_check, mock_parse_args):
        """Test that initial check exceptions are handled gracefully."""
        # Setup mocks
        mock_validate.return_value = True
        mock_load.return_value = {
            "model": "test.tflite",
            "target_device": "gpu",
        }

        # Make initial check raise an exception
        mock_initial_check.side_effect = Exception("Device spec not found")

        # Create args
        mock_args = argparse.Namespace(
            config_path="test.ef",
            check_only=False,  # Not check-only, continue to optimization
            verbose=False,
            dry_run=False,
            docker=False,
            fast_compile=False,
            explain=False,
        )
        mock_parse_args.return_value = mock_args

        # Run main - should handle exception and continue
        with patch("logging.warning") as mock_warning:
            with patch("edgeflowc.create_program_from_dict") as mock_create:
                with patch("edgeflowc.IRBuilder"):
                    with patch("edgeflowc.apply_ir_transformations"):
                        with patch("edgeflowc.CodeGenerator"):
                            with patch("edgeflowc.optimize_model") as mock_opt:
                                with patch("os.makedirs"):
                                    with patch("builtins.open", create=True):
                                        mock_create.return_value = Mock(statements=[])
                                        mock_opt.return_value = (True, "Success", {})

                                        result = edgeflowc.main()

                                        # Should log warning but continue
                                        mock_warning.assert_called()
                                        assert "Initial check failed" in mock_warning.call_args[0][0]

        assert result == 0

    @patch("edgeflowc.parse_arguments")
    @patch("initial_check.perform_initial_check")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_initial_check_skip_optimization(self, mock_validate, mock_load, mock_initial_check, mock_parse_args):
        """Test that optimization is skipped when model already fits."""
        # Setup mocks
        mock_validate.return_value = True
        mock_load.return_value = {
            "model": "already_optimized.tflite",
            "target_device": "cpu",
        }

        mock_report = Mock()
        mock_report.estimated_fit_score = 100.0
        mock_report.issues = []
        mock_report.recommendations = []
        # should_optimize=False means skip optimization
        mock_initial_check.return_value = (False, mock_report)

        # Create args
        mock_args = argparse.Namespace(
            config_path="test.ef",
            check_only=False,  # Not check-only, but will skip due to fit
            verbose=False,
            dry_run=False,
            docker=False,
            fast_compile=False,
            explain=False,
            device_spec_file=None,
        )
        mock_parse_args.return_value = mock_args

        # Run main - should return early without optimization
        with patch("edgeflowc.optimize_model") as mock_optimize:
            result = edgeflowc.main()
            # Should NOT call optimize since model already fits
            mock_optimize.assert_not_called()

        assert result == 0

    @patch("edgeflowc.parse_arguments")
    @patch("initial_check.perform_initial_check")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_check_with_device_spec_file(self, mock_validate, mock_load, mock_initial_check, mock_parse_args):
        """Test initial check with custom device spec file."""
        # Setup mocks
        mock_validate.return_value = True
        mock_load.return_value = {
            "model": "test.tflite",
            "target_device": "custom_device",
        }

        mock_report = Mock()
        mock_report.estimated_fit_score = 75.0
        mock_report.issues = ["Custom device constraint violated"]
        mock_report.recommendations = []
        mock_initial_check.return_value = (True, mock_report)

        # Create args with device_spec_file
        mock_args = argparse.Namespace(
            config_path="test.ef",
            check_only=True,
            verbose=False,
            dry_run=False,
            docker=False,
            fast_compile=False,
            explain=False,
            device_spec_file="custom_device.json",
        )
        mock_parse_args.return_value = mock_args

        # Run main
        result = edgeflowc.main()

        # Verify device spec was passed
        assert result == 0
        mock_initial_check.assert_called_once_with(
            "test.tflite", mock_load.return_value, "custom_device.json"
        )