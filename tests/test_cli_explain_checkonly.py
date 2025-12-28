import argparse
import sys
from unittest.mock import Mock, patch

import edgeflowc


class TestExplainAndCheckOnly:
    @patch("edgeflowc.parse_args")
    @patch("edgeflowc.explain_report")
    @patch("edgeflowc.optimize_model")
    @patch("edgeflowc.codegen_tflite")
    @patch("edgeflowc.apply_ir_optimizations")
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
        mock_codegen.return_value = True
        mock_optimize.return_value = ("optimized.tflite", {"size_reduction": 0.5})

        # Create args with explain=True
        mock_args = argparse.Namespace(
            config_path="test.ef",
            check_only=False,
            verbose=False,
            dry_run=False,
            docker=False,
            fast_compile=False,
            explain=True,
            device_spec_file=None,
        )
        mock_parse_args.return_value = mock_args

        # Run main
        result = edgeflowc.main()

        # Verify explain_report was called
        assert result == 0
        mock_explain_report.assert_called_once()
        args, kwargs = mock_explain_report.call_args
        assert args[0] == mock_ir_graph
        assert args[1] == {"passes_applied": 5}
        assert args[2] == {"size_reduction": 0.5}

    @patch("edgeflowc.parse_args")
    @patch("edgeflowc.initial_check")
    @patch("edgeflowc.load_config")
    @patch("edgeflowc.validate_file_path")
    def test_check_only_flag(
        self, mock_validate, mock_load, mock_initial_check, mock_parse_args
    ):
        """Test --check-only flag stops after initial check."""
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
