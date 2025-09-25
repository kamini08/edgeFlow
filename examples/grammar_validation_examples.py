"""
Examples demonstrating grammar-based type constraints and validation.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgeflow_ast import (
    ActivationType,
    ConstrainedInt,
    Conv2DDeclaration,
    DenseDeclaration,
    DropoutDeclaration,
    DropoutRate,
    KernelSize,
    LayerType,
    PaddingType,
    Program,
    StrideValue,
)
from grammar.grammar_validator import (
    GrammarSemanticValidator,
    create_conv2d_from_params,
    create_dense_from_params,
    create_dropout_from_params,
    validate_grammar_and_semantics,
)
from semantic_analyzer.constraints import (
    get_edge_device_config,
    get_mobile_device_config,
)


def example_1_valid_layer_creation():
    """Example 1: Creating valid layers with type constraints."""
    print("🔍 Example 1: Valid Layer Creation with Type Constraints")
    print("=" * 60)

    try:
        # Create valid Conv2D layer
        conv_params = {
            "filters": 32,
            "kernel_size": (3, 3),
            "strides": (1, 1),
            "padding": "valid",
            "activation": "relu",
            "use_bias": True,
        }

        conv_layer = create_conv2d_from_params("conv1", conv_params, line=1, column=1)
        print(f"✅ Created Conv2D layer: {conv_layer.name}")
        print(f"   Filters: {conv_layer.filters.value}")
        print(f"   Kernel size: {conv_layer.kernel_size.size}")
        print(f"   Activation: {conv_layer.activation.value}")

        # Create valid Dense layer
        dense_params = {"units": 128, "activation": "relu", "use_bias": True}

        dense_layer = create_dense_from_params("dense1", dense_params, line=2, column=1)
        print(f"✅ Created Dense layer: {dense_layer.name}")
        print(f"   Units: {dense_layer.units.value}")
        print(f"   Activation: {dense_layer.activation.value}")

        # Create valid Dropout layer
        dropout_params = {"rate": 0.5}
        dropout_layer = create_dropout_from_params(
            "dropout1", dropout_params, line=3, column=1
        )
        print(f"✅ Created Dropout layer: {dropout_layer.name}")
        print(f"   Rate: {dropout_layer.rate.rate}")

        return True

    except Exception as e:
        print(f"❌ Error creating layers: {e}")
        return False


def example_2_invalid_parameter_values():
    """Example 2: Attempting to create layers with invalid parameter values."""
    print("\n🔍 Example 2: Invalid Parameter Values (Grammar Constraints)")
    print("=" * 60)

    test_cases = [
        {
            "name": "Conv2D with invalid kernel size",
            "func": create_conv2d_from_params,
            "args": (
                "conv_bad",
                {"filters": 32, "kernel_size": 20},
                1,
                1,
            ),  # kernel_size > 15
            "expected_error": "kernel size",
        },
        {
            "name": "Conv2D with invalid stride",
            "func": create_conv2d_from_params,
            "args": (
                "conv_bad2",
                {"filters": 32, "kernel_size": 3, "strides": 10},
                2,
                1,
            ),  # stride > 8
            "expected_error": "stride",
        },
        {
            "name": "Dense with zero units",
            "func": create_dense_from_params,
            "args": ("dense_bad", {"units": 0}, 3, 1),  # units must be positive
            "expected_error": "units",
        },
        {
            "name": "Dropout with invalid rate",
            "func": create_dropout_from_params,
            "args": ("dropout_bad", {"rate": 1.5}, 4, 1),  # rate > 0.9
            "expected_error": "rate",
        },
    ]

    for test_case in test_cases:
        try:
            test_case["func"](*test_case["args"])
            print(f"❌ {test_case['name']}: Should have failed but didn't!")
        except ValueError as e:
            if test_case["expected_error"].lower() in str(e).lower():
                print(f"✅ {test_case['name']}: Correctly caught error")
                print(f"   Error: {e}")
            else:
                print(f"⚠️  {test_case['name']}: Caught error but wrong type")
                print(f"   Error: {e}")
        except Exception as e:
            print(f"❌ {test_case['name']}: Unexpected error type")
            print(f"   Error: {e}")


def example_3_grammar_validation():
    """Example 3: Full grammar validation with layer sequences."""
    print("\n🔍 Example 3: Grammar Validation with Layer Sequences")
    print("=" * 60)

    # Create a program with layers
    layers = []

    # Valid layers
    try:
        conv1 = create_conv2d_from_params(
            "conv1",
            {"filters": 32, "kernel_size": 3, "activation": "relu"},
            line=1,
            column=1,
        )
        layers.append(conv1)

        # This will create a forbidden sequence (Conv2D -> Dense without Flatten)
        dense1 = create_dense_from_params(
            "dense1", {"units": 128, "activation": "relu"}, line=2, column=1
        )
        layers.append(dense1)

        program = Program(statements=layers)

        # Run grammar validation
        config = get_edge_device_config()
        validator = GrammarSemanticValidator(config)
        errors = validator.validate_ast(program)

        print(f"Validation completed:")
        if errors.has_errors():
            print(f"❌ Found {len(errors.errors)} errors:")
            for error in errors.errors:
                print(f"   • {error}")
        else:
            print("✅ No errors found")

        if len(errors.get_errors_by_severity(errors.ErrorSeverity.WARNING)) > 0:
            warnings = errors.get_errors_by_severity(errors.ErrorSeverity.WARNING)
            print(f"⚠️  Found {len(warnings)} warnings:")
            for warning in warnings:
                print(f"   • {warning}")

    except Exception as e:
        print(f"❌ Error during validation: {e}")


def example_4_device_compatibility():
    """Example 4: Device compatibility validation."""
    print("\n🔍 Example 4: Device Compatibility Validation")
    print("=" * 60)

    # Create layers that exceed device limits
    layers = []

    try:
        # Conv2D with too many filters for edge device
        conv_heavy = create_conv2d_from_params(
            "conv_heavy",
            {
                "filters": 1024,  # Exceeds edge device limit (256)
                "kernel_size": 7,
                "activation": "relu",
            },
            line=1,
            column=1,
        )
        layers.append(conv_heavy)

        # Dense with too many units for edge device
        dense_heavy = create_dense_from_params(
            "dense_heavy",
            {"units": 2048, "activation": "relu"},  # Exceeds edge device limit (1024)
            line=2,
            column=1,
        )
        layers.append(dense_heavy)

        program = Program(statements=layers)

        # Test with edge device config (strict limits)
        print("Testing with Edge Device Config:")
        edge_config = get_edge_device_config()
        validator = GrammarSemanticValidator(edge_config)
        errors = validator.validate_ast(program)

        if errors.has_errors():
            print(f"❌ Edge device validation failed with {len(errors.errors)} errors:")
            for error in errors.errors:
                print(f"   • {error}")

        # Test with mobile device config (more permissive)
        print("\nTesting with Mobile Device Config:")
        mobile_config = get_mobile_device_config()
        validator = GrammarSemanticValidator(mobile_config)
        errors = validator.validate_ast(program)

        if errors.has_errors():
            print(
                f"❌ Mobile device validation failed with {len(errors.errors)} errors:"
            )
            for error in errors.errors:
                print(f"   • {error}")
        else:
            print("✅ Mobile device validation passed")

    except Exception as e:
        print(f"❌ Error during device compatibility test: {e}")


def example_5_comprehensive_validation():
    """Example 5: Comprehensive validation combining grammar and semantic analysis."""
    print("\n🔍 Example 5: Comprehensive Grammar + Semantic Validation")
    print("=" * 60)

    # Create a more complex model
    layers = []

    try:
        # Input layer (would be handled by parser)

        # Conv2D with borderline parameters
        conv1 = create_conv2d_from_params(
            "conv1",
            {
                "filters": 64,
                "kernel_size": (5, 5),  # Larger kernel
                "strides": (2, 2),
                "padding": "same",
                "activation": "relu",
            },
            line=1,
            column=1,
        )
        layers.append(conv1)

        # Another Conv2D
        conv2 = create_conv2d_from_params(
            "conv2",
            {"filters": 128, "kernel_size": 3, "activation": "relu"},
            line=2,
            column=1,
        )
        layers.append(conv2)

        # Dropout with high rate
        dropout1 = create_dropout_from_params(
            "dropout1",
            {"rate": 0.8},  # High dropout rate - should generate warning
            line=3,
            column=1,
        )
        layers.append(dropout1)

        # Dense layer (this creates forbidden sequence without Flatten)
        dense1 = create_dense_from_params(
            "dense1", {"units": 256, "activation": "relu"}, line=4, column=1
        )
        layers.append(dense1)

        program = Program(statements=layers)

        # Run comprehensive validation
        config = get_edge_device_config()
        errors = validate_grammar_and_semantics(program, config)

        print("Comprehensive validation results:")
        errors.print_summary()

    except Exception as e:
        print(f"❌ Error during comprehensive validation: {e}")


def run_all_grammar_examples():
    """Run all grammar validation examples."""
    print("🚀 Grammar-Based Type Constraints & Validation Examples")
    print("=" * 70)

    examples = [
        ("Valid Layer Creation", example_1_valid_layer_creation),
        ("Invalid Parameter Values", example_2_invalid_parameter_values),
        ("Grammar Validation", example_3_grammar_validation),
        ("Device Compatibility", example_4_device_compatibility),
        ("Comprehensive Validation", example_5_comprehensive_validation),
    ]

    results = []
    for name, example_func in examples:
        try:
            success = example_func()
            if success is None:  # Some examples don't return success status
                success = True
            results.append((name, "✅ COMPLETED" if success else "❌ FAILED"))
        except Exception as e:
            results.append((name, f"💥 ERROR: {str(e)}"))

    print(f"\n📋 Summary of Grammar Examples")
    print("=" * 70)
    for name, result in results:
        print(f"{result:<15} {name}")

    print(f"\n🎯 Key Benefits Demonstrated:")
    print(f"  • Early error detection at grammar level")
    print(f"  • Type-safe parameter validation")
    print(f"  • Enumerated type constraints (activation, padding)")
    print(f"  • Range-constrained numeric values")
    print(f"  • Device-specific compatibility checking")
    print(f"  • Clear, actionable error messages with source locations")


if __name__ == "__main__":
    run_all_grammar_examples()
