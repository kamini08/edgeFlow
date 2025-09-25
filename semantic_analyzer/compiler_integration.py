"""
Integration module for plugging semantic analyzer into DSL compiler pipeline.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analyzer import SemanticAnalyzer
from .constraints import (
    ConstraintConfig,
    get_edge_device_config,
    get_mobile_device_config,
    get_server_device_config,
)
from .error_types import ErrorCollector, ErrorSeverity
from .ir_nodes import IRGraph


class CompilerPipeline:
    """Main compiler pipeline that integrates semantic analysis."""

    def __init__(self, target_device: str = "edge", config_path: Optional[str] = None):
        self.target_device = target_device
        self.config = self._load_config(config_path)
        self.semantic_analyzer = SemanticAnalyzer(self.config)
        self.compilation_context = {}

    def _load_config(self, config_path: Optional[str]) -> ConstraintConfig:
        """Load configuration based on target device or config file."""
        if config_path and Path(config_path).exists():
            return self._load_config_from_file(config_path)

        # Use predefined configurations
        device_configs = {
            "edge": get_edge_device_config,
            "mobile": get_mobile_device_config,
            "server": get_server_device_config,
            "cloud": get_server_device_config,
        }

        config_func = device_configs.get(
            self.target_device.lower(), get_edge_device_config
        )
        return config_func()

    def _load_config_from_file(self, config_path: str) -> ConstraintConfig:
        """Load configuration from JSON file."""
        # This would implement loading from a configuration file
        # For now, return default config
        return ConstraintConfig()

    def compile_dsl(
        self, dsl_source: str, source_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete compilation pipeline from DSL source to validated IR.

        Args:
            dsl_source: The DSL source code
            source_file: Optional source file path for error reporting

        Returns:
            Dictionary containing compilation results and any errors
        """
        result = {
            "success": False,
            "ir_graph": None,
            "errors": [],
            "warnings": [],
            "metadata": {},
        }

        try:
            # Phase 1: Parse DSL to IR (placeholder - would use actual parser)
            ir_graph = self._parse_dsl_to_ir(dsl_source, source_file)
            if not ir_graph:
                result["errors"].append("Failed to parse DSL source")
                return result

            # Phase 2: Semantic Analysis
            error_collector = self.semantic_analyzer.analyze(ir_graph)

            # Process results
            result["ir_graph"] = ir_graph
            result["errors"] = [
                str(error)
                for error in error_collector.get_errors_by_severity(ErrorSeverity.ERROR)
            ]
            result["errors"].extend(
                [
                    str(error)
                    for error in error_collector.get_errors_by_severity(
                        ErrorSeverity.FATAL
                    )
                ]
            )
            result["warnings"] = [
                str(error)
                for error in error_collector.get_errors_by_severity(
                    ErrorSeverity.WARNING
                )
            ]

            # Determine success
            result["success"] = not error_collector.has_errors()

            # Add metadata
            result["metadata"] = {
                "target_device": self.target_device,
                "analysis_summary": self.semantic_analyzer.get_analysis_summary(),
                "graph_stats": self._get_graph_statistics(ir_graph),
            }

            # Phase 3: Code generation (if no blocking errors)
            if result["success"]:
                generated_code = self._generate_code(ir_graph)
                result["generated_code"] = generated_code

        except Exception as e:
            result["errors"].append(f"Compilation failed: {str(e)}")

        return result

    def _parse_dsl_to_ir(
        self, dsl_source: str, source_file: Optional[str]
    ) -> Optional[IRGraph]:
        """
        Parse DSL source to IR graph.
        This is a placeholder - in a real implementation, this would use
        your actual DSL parser (ANTLR, Lark, PLY, etc.)
        """
        # Placeholder implementation - would be replaced with actual parser
        # For demonstration, create a simple example graph
        return self._create_example_ir_graph()

    def _create_example_ir_graph(self) -> IRGraph:
        """Create an example IR graph for demonstration."""
        from .ir_nodes import (
            ActivationType,
            DataType,
            IRNode,
            LayerType,
            TensorShape,
            create_conv2d_node,
            create_dense_node,
            create_input_node,
        )

        graph = IRGraph()

        # Input layer
        input_node = create_input_node(
            node_id="input_1",
            shape=TensorShape((28, 28, 1)),
            dtype=DataType.FLOAT32,
            name="input",
        )
        graph.add_node(input_node)

        # Conv2D layer
        conv_node = create_conv2d_node(
            node_id="conv_1",
            filters=32,
            kernel_size=(3, 3),
            activation=ActivationType.RELU,
            name="conv2d",
        )
        graph.add_node(conv_node)

        # Flatten layer
        flatten_node = IRNode(
            node_id="flatten_1", layer_type=LayerType.FLATTEN, name="flatten"
        )
        graph.add_node(flatten_node)

        # Dense layer
        dense_node = create_dense_node(
            node_id="dense_1", units=128, activation=ActivationType.RELU, name="dense"
        )
        graph.add_node(dense_node)

        # Output layer
        output_node = create_dense_node(
            node_id="output_1",
            units=10,
            activation=ActivationType.SOFTMAX,
            name="output",
        )
        output_node.layer_type = LayerType.OUTPUT
        graph.add_node(output_node)

        # Connect nodes
        input_node.connect_to(conv_node)
        conv_node.connect_to(flatten_node)
        flatten_node.connect_to(dense_node)
        dense_node.connect_to(output_node)

        return graph

    def _generate_code(self, ir_graph: IRGraph) -> Dict[str, str]:
        """
        Generate target code from validated IR.
        This is a placeholder for actual code generation.
        """
        # Placeholder implementation
        return {
            "framework": "tensorflow",
            "code": "# Generated TensorFlow code would go here",
            "optimization_level": "O2",
        }

    def _get_graph_statistics(self, ir_graph: IRGraph) -> Dict[str, Any]:
        """Get statistics about the IR graph."""
        layer_counts = {}
        for node in ir_graph.nodes.values():
            layer_type = node.layer_type.value
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1

        return {
            "total_nodes": len(ir_graph),
            "total_connections": sum(
                len(node.output_nodes) for node in ir_graph.nodes.values()
            ),
            "layer_counts": layer_counts,
            "memory_usage_mb": ir_graph.calculate_total_memory_usage() / (1024 * 1024),
            "has_cycles": ir_graph.has_cycles(),
            "is_connected": ir_graph.is_connected(),
        }

    def validate_only(self, ir_graph: IRGraph) -> ErrorCollector:
        """Perform only semantic validation without full compilation."""
        return self.semantic_analyzer.analyze(ir_graph)

    def set_target_device(self, device: str) -> None:
        """Change target device and reload configuration."""
        self.target_device = device
        self.config = self._load_config(None)
        self.semantic_analyzer = SemanticAnalyzer(self.config)


class CLIInterface:
    """Command-line interface for the semantic analyzer."""

    def __init__(self):
        self.pipeline = None

    def run_analysis(
        self,
        dsl_file: str,
        target_device: str = "edge",
        config_file: Optional[str] = None,
        output_format: str = "text",
    ) -> None:
        """Run semantic analysis from command line."""
        try:
            # Initialize pipeline
            self.pipeline = CompilerPipeline(target_device, config_file)

            # Read DSL file
            with open(dsl_file, "r") as f:
                dsl_source = f.read()

            # Compile and analyze
            result = self.pipeline.compile_dsl(dsl_source, dsl_file)

            # Output results
            if output_format == "json":
                self._output_json(result)
            else:
                self._output_text(result)

        except FileNotFoundError:
            print(f"❌ Error: DSL file '{dsl_file}' not found")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    def _output_text(self, result: Dict[str, Any]) -> None:
        """Output results in human-readable text format."""
        print("🔍 Semantic Analysis Results")
        print("=" * 50)

        if result["success"]:
            print("✅ Analysis passed - no blocking errors found")
        else:
            print("❌ Analysis failed - blocking errors found")

        # Print errors
        if result["errors"]:
            print(f"\n🚨 Errors ({len(result['errors'])}):")
            for error in result["errors"]:
                print(f"  • {error}")

        # Print warnings
        if result["warnings"]:
            print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
            for warning in result["warnings"]:
                print(f"  • {warning}")

        # Print metadata
        if "metadata" in result:
            metadata = result["metadata"]
            print(f"\n📊 Analysis Summary:")
            print(f"  Target Device: {metadata['target_device']}")

            if "graph_stats" in metadata:
                stats = metadata["graph_stats"]
                print(f"  Graph Nodes: {stats['total_nodes']}")
                print(f"  Memory Usage: {stats['memory_usage_mb']:.2f} MB")
                print(f"  Layer Types: {', '.join(stats['layer_counts'].keys())}")

    def _output_json(self, result: Dict[str, Any]) -> None:
        """Output results in JSON format."""
        # Remove non-serializable objects
        output = {
            "success": result["success"],
            "errors": result["errors"],
            "warnings": result["warnings"],
            "metadata": result.get("metadata", {}),
        }
        print(json.dumps(output, indent=2))


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="DSL Semantic Analyzer")
    parser.add_argument("dsl_file", help="Path to DSL source file")
    parser.add_argument(
        "--device",
        default="edge",
        choices=["edge", "mobile", "server", "cloud"],
        help="Target device type",
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument(
        "--format", default="text", choices=["text", "json"], help="Output format"
    )

    args = parser.parse_args()

    cli = CLIInterface()
    cli.run_analysis(args.dsl_file, args.device, args.config, args.format)


if __name__ == "__main__":
    main()
