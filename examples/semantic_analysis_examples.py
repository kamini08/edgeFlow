"""
Comprehensive examples demonstrating the semantic analyzer capabilities.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_analyzer import (
    ActivationType,
    DataType,
    IRGraph,
    IRNode,
    LayerType,
    SemanticAnalyzer,
    SourceLocation,
    TensorInfo,
    TensorShape,
    create_conv2d_node,
    create_dense_node,
    create_input_node,
    get_edge_device_config,
    get_mobile_device_config,
)


def example_1_valid_cnn_model():
    """Example 1: Valid CNN model that should pass all checks."""
    print("🔍 Example 1: Valid CNN Model")
    print("=" * 50)

    # Create a valid CNN architecture
    graph = IRGraph()

    # Input layer: 28x28x1 (MNIST-like)
    input_node = create_input_node(
        node_id="input_1",
        shape=TensorShape((28, 28, 1)),
        dtype=DataType.FLOAT32,
        name="input",
        location=SourceLocation(line=1, column=1, file_path="model.dsl"),
    )
    input_node.add_output_tensor(
        TensorInfo(
            shape=TensorShape((28, 28, 1)), dtype=DataType.FLOAT32, name="input_tensor"
        )
    )
    graph.add_node(input_node)

    # Conv2D layer: 32 filters, 3x3 kernel
    conv_node = create_conv2d_node(
        node_id="conv_1",
        filters=32,
        kernel_size=(3, 3),
        activation=ActivationType.RELU,
        name="conv2d_1",
        location=SourceLocation(line=2, column=1, file_path="model.dsl"),
    )
    conv_node.add_input_tensor(
        TensorInfo(shape=TensorShape((28, 28, 1)), dtype=DataType.FLOAT32)
    )
    conv_node.add_output_tensor(
        TensorInfo(
            shape=TensorShape((26, 26, 32)),  # Valid padding: 28-3+1=26
            dtype=DataType.FLOAT32,
        )
    )
    graph.add_node(conv_node)

    # MaxPool2D layer
    pool_node = IRNode(
        node_id="pool_1",
        layer_type=LayerType.MAXPOOL2D,
        name="maxpool2d_1",
        parameters={"pool_size": (2, 2), "strides": (2, 2), "padding": "valid"},
        location=SourceLocation(line=3, column=1, file_path="model.dsl"),
    )
    pool_node.add_input_tensor(
        TensorInfo(shape=TensorShape((26, 26, 32)), dtype=DataType.FLOAT32)
    )
    pool_node.add_output_tensor(
        TensorInfo(shape=TensorShape((13, 13, 32)), dtype=DataType.FLOAT32)  # 26/2 = 13
    )
    graph.add_node(pool_node)

    # Flatten layer
    flatten_node = IRNode(
        node_id="flatten_1",
        layer_type=LayerType.FLATTEN,
        name="flatten_1",
        location=SourceLocation(line=4, column=1, file_path="model.dsl"),
    )
    flatten_node.add_input_tensor(
        TensorInfo(shape=TensorShape((13, 13, 32)), dtype=DataType.FLOAT32)
    )
    flatten_node.add_output_tensor(
        TensorInfo(
            shape=TensorShape((5408,)), dtype=DataType.FLOAT32  # 13*13*32 = 5408
        )
    )
    graph.add_node(flatten_node)

    # Dense layer
    dense_node = create_dense_node(
        node_id="dense_1",
        units=128,
        activation=ActivationType.RELU,
        name="dense_1",
        location=SourceLocation(line=5, column=1, file_path="model.dsl"),
    )
    dense_node.add_input_tensor(
        TensorInfo(shape=TensorShape((5408,)), dtype=DataType.FLOAT32)
    )
    dense_node.add_output_tensor(
        TensorInfo(shape=TensorShape((128,)), dtype=DataType.FLOAT32)
    )
    graph.add_node(dense_node)

    # Output layer
    output_node = create_dense_node(
        node_id="output_1",
        units=10,
        activation=ActivationType.SOFTMAX,
        name="output",
        location=SourceLocation(line=6, column=1, file_path="model.dsl"),
    )
    output_node.layer_type = LayerType.OUTPUT
    output_node.add_input_tensor(
        TensorInfo(shape=TensorShape((128,)), dtype=DataType.FLOAT32)
    )
    output_node.add_output_tensor(
        TensorInfo(shape=TensorShape((10,)), dtype=DataType.FLOAT32)
    )
    graph.add_node(output_node)

    # Connect the nodes
    input_node.connect_to(conv_node)
    conv_node.connect_to(pool_node)
    pool_node.connect_to(flatten_node)
    flatten_node.connect_to(dense_node)
    dense_node.connect_to(output_node)

    # Analyze with mobile device config (more permissive than edge)
    config = get_mobile_device_config()
    analyzer = SemanticAnalyzer(config)
    errors = analyzer.analyze(graph)

    print(f"Analysis completed:")
    errors.print_summary()
    return errors.has_errors()


def example_2_shape_mismatch_errors():
    """Example 2: Model with shape mismatch errors."""
    print("\n🔍 Example 2: Shape Mismatch Errors")
    print("=" * 50)

    graph = IRGraph()

    # Input layer
    input_node = create_input_node(
        node_id="input_1",
        shape=TensorShape((28, 28, 1)),
        dtype=DataType.FLOAT32,
        location=SourceLocation(line=1, column=1, file_path="bad_model.dsl"),
    )
    input_node.add_output_tensor(
        TensorInfo(shape=TensorShape((28, 28, 1)), dtype=DataType.FLOAT32)
    )
    graph.add_node(input_node)

    # Dense layer expecting wrong input shape (should be flattened)
    dense_node = create_dense_node(
        node_id="dense_1",
        units=128,
        name="dense_bad",
        location=SourceLocation(line=2, column=1, file_path="bad_model.dsl"),
    )
    # Intentionally wrong input shape - Dense expects 2D, getting 4D
    dense_node.add_input_tensor(
        TensorInfo(
            shape=TensorShape((32, 32, 3)), dtype=DataType.FLOAT32  # Wrong shape!
        )
    )
    dense_node.add_output_tensor(
        TensorInfo(shape=TensorShape((128,)), dtype=DataType.FLOAT32)
    )
    graph.add_node(dense_node)

    # Connect with mismatched shapes
    input_node.connect_to(dense_node)

    # Analyze
    analyzer = SemanticAnalyzer()
    errors = analyzer.analyze(graph)

    print(f"Analysis completed:")
    errors.print_summary()
    return errors.has_errors()


def example_3_parameter_range_errors():
    """Example 3: Model with parameter range violations."""
    print("\n🔍 Example 3: Parameter Range Errors")
    print("=" * 50)

    graph = IRGraph()

    # Input layer
    input_node = create_input_node(
        node_id="input_1",
        shape=TensorShape((224, 224, 3)),
        dtype=DataType.FLOAT32,
        location=SourceLocation(line=1, column=1, file_path="param_error.dsl"),
    )
    graph.add_node(input_node)

    # Conv2D with invalid parameters
    conv_node = IRNode(
        node_id="conv_bad",
        layer_type=LayerType.CONV2D,
        name="conv_bad",
        parameters={
            "filters": 2048,  # Too many filters for edge device
            "kernel_size": (15, 15),  # Kernel too large
            "strides": (10, 10),  # Strides too large
            "activation": "invalid_activation",  # Invalid activation
            "padding": "invalid_padding",  # Invalid padding
        },
        location=SourceLocation(line=2, column=1, file_path="param_error.dsl"),
    )
    graph.add_node(conv_node)

    # Dense layer with too many units
    dense_node = IRNode(
        node_id="dense_bad",
        layer_type=LayerType.DENSE,
        name="dense_bad",
        parameters={
            "units": 50000,  # Way too many units
            "activation": ActivationType.RELU,
        },
        location=SourceLocation(line=3, column=1, file_path="param_error.dsl"),
    )
    graph.add_node(dense_node)

    # Dropout with invalid rate
    dropout_node = IRNode(
        node_id="dropout_bad",
        layer_type=LayerType.DROPOUT,
        name="dropout_bad",
        parameters={"rate": 1.5},  # Rate > 1.0 is invalid
        location=SourceLocation(line=4, column=1, file_path="param_error.dsl"),
    )
    graph.add_node(dropout_node)

    # Connect nodes
    input_node.connect_to(conv_node)
    conv_node.connect_to(dense_node)
    dense_node.connect_to(dropout_node)

    # Analyze with edge device config (strict limits)
    config = get_edge_device_config()
    analyzer = SemanticAnalyzer(config)
    errors = analyzer.analyze(graph)

    print(f"Analysis completed:")
    errors.print_summary()
    return errors.has_errors()


def example_4_forbidden_sequences():
    """Example 4: Model with forbidden layer sequences."""
    print("\n🔍 Example 4: Forbidden Layer Sequences")
    print("=" * 50)

    graph = IRGraph()

    # Input layer
    input_node = create_input_node(
        node_id="input_1",
        shape=TensorShape((28, 28, 1)),
        dtype=DataType.FLOAT32,
        location=SourceLocation(line=1, column=1, file_path="sequence_error.dsl"),
    )
    input_node.add_output_tensor(
        TensorInfo(shape=TensorShape((28, 28, 1)), dtype=DataType.FLOAT32)
    )
    graph.add_node(input_node)

    # Conv2D layer
    conv_node = create_conv2d_node(
        node_id="conv_1",
        filters=32,
        kernel_size=(3, 3),
        location=SourceLocation(line=2, column=1, file_path="sequence_error.dsl"),
    )
    conv_node.add_input_tensor(
        TensorInfo(shape=TensorShape((28, 28, 1)), dtype=DataType.FLOAT32)
    )
    conv_node.add_output_tensor(
        TensorInfo(shape=TensorShape((26, 26, 32)), dtype=DataType.FLOAT32)
    )
    graph.add_node(conv_node)

    # Dense layer directly after Conv2D (forbidden without Flatten)
    dense_node = create_dense_node(
        node_id="dense_1",
        units=128,
        location=SourceLocation(line=3, column=1, file_path="sequence_error.dsl"),
    )
    dense_node.add_input_tensor(
        TensorInfo(
            shape=TensorShape((26, 26, 32)),  # 4D tensor to Dense (wrong!)
            dtype=DataType.FLOAT32,
        )
    )
    dense_node.add_output_tensor(
        TensorInfo(shape=TensorShape((128,)), dtype=DataType.FLOAT32)
    )
    graph.add_node(dense_node)

    # Connect nodes (this creates the forbidden sequence)
    input_node.connect_to(conv_node)
    conv_node.connect_to(dense_node)  # Conv2D -> Dense without Flatten

    # Analyze
    analyzer = SemanticAnalyzer()
    errors = analyzer.analyze(graph)

    print(f"Analysis completed:")
    errors.print_summary()
    return errors.has_errors()


def example_5_resource_constraints():
    """Example 5: Model exceeding resource constraints."""
    print("\n🔍 Example 5: Resource Constraint Violations")
    print("=" * 50)

    graph = IRGraph()

    # Create a model that exceeds edge device memory limits
    input_node = create_input_node(
        node_id="input_1",
        shape=TensorShape((1024, 1024, 3)),  # Very large input
        dtype=DataType.FLOAT32,
        location=SourceLocation(line=1, column=1, file_path="resource_heavy.dsl"),
    )
    input_node.add_output_tensor(
        TensorInfo(shape=TensorShape((1024, 1024, 3)), dtype=DataType.FLOAT32)
    )
    graph.add_node(input_node)

    # Multiple large Conv2D layers
    prev_node = input_node
    prev_shape = TensorShape((1024, 1024, 3))

    for i in range(3):
        conv_node = create_conv2d_node(
            node_id=f"conv_{i+1}",
            filters=512,  # Many filters
            kernel_size=(7, 7),  # Large kernels
            padding="same",
            location=SourceLocation(
                line=i + 2, column=1, file_path="resource_heavy.dsl"
            ),
        )
        conv_node.add_input_tensor(TensorInfo(shape=prev_shape, dtype=DataType.FLOAT32))
        output_shape = TensorShape(
            (prev_shape.dimensions[0], prev_shape.dimensions[1], 512)
        )
        conv_node.add_output_tensor(
            TensorInfo(shape=output_shape, dtype=DataType.FLOAT32)
        )
        graph.add_node(conv_node)

        prev_node.connect_to(conv_node)
        prev_node = conv_node
        prev_shape = output_shape

    # Analyze with edge device config (strict memory limits)
    config = get_edge_device_config()
    analyzer = SemanticAnalyzer(config)
    errors = analyzer.analyze(graph)

    print(f"Analysis completed:")
    errors.print_summary()

    # Show memory usage
    total_memory_mb = graph.calculate_total_memory_usage() / (1024 * 1024)
    print(f"\n📊 Memory Analysis:")
    print(f"  Total memory usage: {total_memory_mb:.2f} MB")
    print(f"  Edge device limit: {config.device_constraints.max_memory_mb} MB")
    print(
        f"  Exceeds limit: {'Yes' if total_memory_mb > config.device_constraints.max_memory_mb else 'No'}"
    )

    return errors.has_errors()


def example_6_device_compatibility():
    """Example 6: Device compatibility issues."""
    print("\n🔍 Example 6: Device Compatibility Issues")
    print("=" * 50)

    graph = IRGraph()

    # Input with unsupported data type for edge device
    input_node = create_input_node(
        node_id="input_1",
        shape=TensorShape((224, 224, 3)),
        dtype=DataType.FLOAT32,  # Edge device might prefer FLOAT16 or INT8
        location=SourceLocation(line=1, column=1, file_path="device_compat.dsl"),
    )
    input_node.add_output_tensor(
        TensorInfo(shape=TensorShape((224, 224, 3)), dtype=DataType.FLOAT32)
    )
    graph.add_node(input_node)

    # Use layers that might not be supported on edge devices
    attention_node = IRNode(
        node_id="attention_1",
        layer_type=LayerType.ATTENTION,
        name="attention_layer",
        parameters={"num_heads": 8, "key_dim": 64},
        location=SourceLocation(line=2, column=1, file_path="device_compat.dsl"),
    )
    attention_node.add_input_tensor(
        TensorInfo(shape=TensorShape((224, 224, 3)), dtype=DataType.FLOAT32)
    )
    attention_node.add_output_tensor(
        TensorInfo(shape=TensorShape((224, 224, 3)), dtype=DataType.FLOAT32)
    )
    graph.add_node(attention_node)

    input_node.connect_to(attention_node)

    # Analyze with edge device config
    config = get_edge_device_config()
    # Modify config to be more restrictive
    config.device_constraints.supported_dtypes = [
        DataType.FLOAT16,
        DataType.INT8,
        DataType.UINT8,
    ]
    config.device_constraints.supported_layers = [
        LayerType.INPUT,
        LayerType.CONV2D,
        LayerType.DENSE,
        LayerType.MAXPOOL2D,
        LayerType.FLATTEN,
        LayerType.OUTPUT,
    ]

    analyzer = SemanticAnalyzer(config)
    errors = analyzer.analyze(graph)

    print(f"Analysis completed:")
    errors.print_summary()
    return errors.has_errors()


def run_all_examples():
    """Run all examples and show summary."""
    print("🚀 Running Semantic Analysis Examples")
    print("=" * 60)

    examples = [
        ("Valid CNN Model", example_1_valid_cnn_model),
        ("Shape Mismatch Errors", example_2_shape_mismatch_errors),
        ("Parameter Range Errors", example_3_parameter_range_errors),
        ("Forbidden Sequences", example_4_forbidden_sequences),
        ("Resource Constraints", example_5_resource_constraints),
        ("Device Compatibility", example_6_device_compatibility),
    ]

    results = []
    for name, example_func in examples:
        try:
            has_errors = example_func()
            results.append((name, "❌ FAILED" if has_errors else "✅ PASSED"))
        except Exception as e:
            results.append((name, f"💥 ERROR: {str(e)}"))

    print("\n📋 Summary of Examples")
    print("=" * 60)
    for name, result in results:
        print(f"{result:<12} {name}")

    print(f"\n🎯 Expected Results:")
    print(f"  • Valid CNN Model should PASS (no errors)")
    print(f"  • All other examples should FAIL (demonstrate error detection)")


if __name__ == "__main__":
    run_all_examples()
