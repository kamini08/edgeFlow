"""
Simple demonstration of grammar-based type constraints and validation.
This shows the core concepts without complex inheritance issues.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

# ============================================================================
# Type Constraint Enums (from our enhanced AST)
# ============================================================================


class LayerType(Enum):
    """Enumeration of supported layer types with validation."""

    CONV2D = "Conv2D"
    DENSE = "Dense"
    DROPOUT = "Dropout"
    MAXPOOL2D = "MaxPool2D"


class ActivationType(Enum):
    """Enumeration of supported activation functions."""

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LINEAR = "linear"


class PaddingType(Enum):
    """Enumeration of supported padding types."""

    VALID = "valid"
    SAME = "same"


# ============================================================================
# Type-Constrained Value Classes
# ============================================================================


@dataclass
class ConstrainedInt:
    """Integer with validation constraints."""

    value: int
    min_val: int = 1
    max_val: int = 10000

    def __post_init__(self):
        if not (self.min_val <= self.value <= self.max_val):
            raise ValueError(
                f"Value {self.value} must be between {self.min_val} and {self.max_val}"
            )


@dataclass
class KernelSize:
    """Kernel size with validation (1-15 for most layers)."""

    size: Union[int, Tuple[int, int]]

    def __post_init__(self):
        if isinstance(self.size, int):
            if not (1 <= self.size <= 15):
                raise ValueError(f"Kernel size {self.size} must be between 1 and 15")
        elif isinstance(self.size, tuple):
            if len(self.size) != 2:
                raise ValueError("Kernel size tuple must have exactly 2 elements")
            for s in self.size:
                if not (1 <= s <= 15):
                    raise ValueError(f"Kernel size {s} must be between 1 and 15")


@dataclass
class StrideValue:
    """Stride value with validation (1-8)."""

    stride: Union[int, Tuple[int, int]]

    def __post_init__(self):
        if isinstance(self.stride, int):
            if not (1 <= self.stride <= 8):
                raise ValueError(f"Stride {self.stride} must be between 1 and 8")
        elif isinstance(self.stride, tuple):
            if len(self.stride) != 2:
                raise ValueError("Stride tuple must have exactly 2 elements")
            for s in self.stride:
                if not (1 <= s <= 8):
                    raise ValueError(f"Stride {s} must be between 1 and 8")


@dataclass
class DropoutRate:
    """Dropout rate with validation (0.0-0.9)."""

    rate: float

    def __post_init__(self):
        if not (0.0 <= self.rate <= 0.9):
            raise ValueError(f"Dropout rate {self.rate} must be between 0.0 and 0.9")


# ============================================================================
# Grammar Validation Functions
# ============================================================================


def validate_conv2d_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Conv2D parameters with grammar-level type constraints.

    This demonstrates how ANTLR grammar rules would validate parameters
    before creating AST nodes.
    """
    validated = {}
    errors = []

    try:
        # Validate filters (positive integer, 1-2048)
        filters_val = params.get("filters")
        if filters_val is None:
            errors.append("Conv2D requires 'filters' parameter")
        else:
            validated["filters"] = ConstrainedInt(filters_val, min_val=1, max_val=2048)

        # Validate kernel_size (1-15, int or tuple)
        kernel_val = params.get("kernel_size")
        if kernel_val is None:
            errors.append("Conv2D requires 'kernel_size' parameter")
        else:
            validated["kernel_size"] = KernelSize(kernel_val)

        # Validate optional parameters
        if "strides" in params:
            validated["strides"] = StrideValue(params["strides"])
        else:
            validated["strides"] = StrideValue(1)  # Default

        if "padding" in params:
            if params["padding"] not in ["valid", "same"]:
                errors.append(
                    f"Invalid padding '{params['padding']}'. Must be 'valid' or 'same'"
                )
            else:
                validated["padding"] = PaddingType(params["padding"])
        else:
            validated["padding"] = PaddingType.VALID  # Default

        if "activation" in params:
            try:
                validated["activation"] = ActivationType(params["activation"])
            except ValueError:
                valid_activations = [a.value for a in ActivationType]
                errors.append(
                    f"Invalid activation '{params['activation']}'. Must be one of: {valid_activations}"
                )
        else:
            validated["activation"] = ActivationType.LINEAR  # Default

        validated["use_bias"] = params.get("use_bias", True)

    except ValueError as e:
        errors.append(str(e))

    if errors:
        raise ValueError(f"Conv2D parameter validation failed: {'; '.join(errors)}")

    return validated


def validate_dense_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Dense layer parameters with type constraints."""
    validated = {}
    errors = []

    try:
        # Validate units (positive integer, 1-10000)
        units_val = params.get("units")
        if units_val is None:
            errors.append("Dense requires 'units' parameter")
        else:
            validated["units"] = ConstrainedInt(units_val, min_val=1, max_val=10000)

        # Validate activation
        if "activation" in params:
            try:
                validated["activation"] = ActivationType(params["activation"])
            except ValueError:
                valid_activations = [a.value for a in ActivationType]
                errors.append(
                    f"Invalid activation '{params['activation']}'. Must be one of: {valid_activations}"
                )
        else:
            validated["activation"] = ActivationType.LINEAR

        validated["use_bias"] = params.get("use_bias", True)

    except ValueError as e:
        errors.append(str(e))

    if errors:
        raise ValueError(f"Dense parameter validation failed: {'; '.join(errors)}")

    return validated


def validate_dropout_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Dropout layer parameters with type constraints."""
    validated = {}
    errors = []

    try:
        # Validate rate (0.0-0.9)
        rate_val = params.get("rate")
        if rate_val is None:
            errors.append("Dropout requires 'rate' parameter")
        else:
            validated["rate"] = DropoutRate(rate_val)

    except ValueError as e:
        errors.append(str(e))

    if errors:
        raise ValueError(f"Dropout parameter validation failed: {'; '.join(errors)}")

    return validated


# ============================================================================
# Device Compatibility Validation
# ============================================================================


@dataclass
class DeviceLimits:
    """Device-specific limits for validation."""

    max_filters: int
    max_kernel_size: int
    max_units: int
    max_memory_mb: float
    supported_dtypes: List[str]


def get_device_limits(device_type: str) -> DeviceLimits:
    """Get device-specific limits."""
    limits = {
        "edge": DeviceLimits(
            max_filters=256,
            max_kernel_size=7,
            max_units=1024,
            max_memory_mb=256.0,
            supported_dtypes=["int8", "uint8", "float16"],
        ),
        "mobile": DeviceLimits(
            max_filters=512,
            max_kernel_size=11,
            max_units=2048,
            max_memory_mb=512.0,
            supported_dtypes=["int8", "uint8", "float16", "float32"],
        ),
        "server": DeviceLimits(
            max_filters=2048,
            max_kernel_size=15,
            max_units=10000,
            max_memory_mb=8192.0,
            supported_dtypes=["int8", "uint8", "int16", "float16", "float32"],
        ),
    }
    return limits.get(device_type, limits["server"])


def validate_device_compatibility(
    layer_type: str, params: Dict[str, Any], device_type: str
) -> List[str]:
    """Validate layer parameters against device constraints."""
    warnings = []
    limits = get_device_limits(device_type)

    if layer_type == "Conv2D":
        if "filters" in params and params["filters"].value > limits.max_filters:
            warnings.append(
                f"Conv2D filters {params['filters'].value} exceeds {device_type} limit ({limits.max_filters})"
            )

        if "kernel_size" in params:
            kernel = params["kernel_size"].size
            max_kernel = kernel if isinstance(kernel, int) else max(kernel)
            if max_kernel > limits.max_kernel_size:
                warnings.append(
                    f"Conv2D kernel size {max_kernel} exceeds {device_type} limit ({limits.max_kernel_size})"
                )

    elif layer_type == "Dense":
        if "units" in params and params["units"].value > limits.max_units:
            warnings.append(
                f"Dense units {params['units'].value} exceeds {device_type} limit ({limits.max_units})"
            )

    return warnings


# ============================================================================
# Demonstration Examples
# ============================================================================


def demo_valid_parameters():
    """Demonstrate valid parameter validation."""
    print("🔍 Demo 1: Valid Parameter Validation")
    print("=" * 50)

    test_cases = [
        {
            "name": "Conv2D with valid parameters",
            "layer_type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": (3, 3),
                "strides": (1, 1),
                "padding": "valid",
                "activation": "relu",
            },
            "validator": validate_conv2d_params,
        },
        {
            "name": "Dense with valid parameters",
            "layer_type": "Dense",
            "params": {"units": 128, "activation": "relu"},
            "validator": validate_dense_params,
        },
        {
            "name": "Dropout with valid parameters",
            "layer_type": "Dropout",
            "params": {"rate": 0.5},
            "validator": validate_dropout_params,
        },
    ]

    for test_case in test_cases:
        try:
            validated = test_case["validator"](test_case["params"])
            print(f"✅ {test_case['name']}: PASSED")

            # Show validated parameters
            for key, value in validated.items():
                if hasattr(value, "value"):
                    print(f"   {key}: {value.value}")
                elif hasattr(value, "size"):
                    print(f"   {key}: {value.size}")
                elif hasattr(value, "stride"):
                    print(f"   {key}: {value.stride}")
                elif hasattr(value, "rate"):
                    print(f"   {key}: {value.rate}")
                elif isinstance(value, Enum):
                    print(f"   {key}: {value.value}")
                else:
                    print(f"   {key}: {value}")

        except Exception as e:
            print(f"❌ {test_case['name']}: FAILED - {e}")


def demo_invalid_parameters():
    """Demonstrate invalid parameter detection."""
    print("\n🔍 Demo 2: Invalid Parameter Detection")
    print("=" * 50)

    test_cases = [
        {
            "name": "Conv2D with invalid kernel size",
            "params": {"filters": 32, "kernel_size": 20},  # > 15
            "validator": validate_conv2d_params,
            "expected_error": "kernel size",
        },
        {
            "name": "Conv2D with invalid stride",
            "params": {"filters": 32, "kernel_size": 3, "strides": 10},  # > 8
            "validator": validate_conv2d_params,
            "expected_error": "stride",
        },
        {
            "name": "Dense with zero units",
            "params": {"units": 0},  # < 1
            "validator": validate_dense_params,
            "expected_error": "units",
        },
        {
            "name": "Dropout with invalid rate",
            "params": {"rate": 1.5},  # > 0.9
            "validator": validate_dropout_params,
            "expected_error": "rate",
        },
        {
            "name": "Conv2D with invalid activation",
            "params": {"filters": 32, "kernel_size": 3, "activation": "invalid"},
            "validator": validate_conv2d_params,
            "expected_error": "activation",
        },
    ]

    for test_case in test_cases:
        try:
            test_case["validator"](test_case["params"])
            print(f"❌ {test_case['name']}: Should have failed but didn't!")
        except ValueError as e:
            if test_case["expected_error"].lower() in str(e).lower():
                print(f"✅ {test_case['name']}: Correctly caught error")
                print(f"   Error: {e}")
            else:
                print(f"⚠️  {test_case['name']}: Caught error but wrong type")
                print(f"   Error: {e}")


def demo_device_compatibility():
    """Demonstrate device compatibility validation."""
    print("\n🔍 Demo 3: Device Compatibility Validation")
    print("=" * 50)

    # Test layer that works on server but not edge
    conv_params = validate_conv2d_params(
        {"filters": 1024, "kernel_size": 7, "activation": "relu"}  # High filter count
    )

    dense_params = validate_dense_params(
        {"units": 4096, "activation": "relu"}  # High unit count
    )

    devices = ["edge", "mobile", "server"]

    for device in devices:
        print(f"\n📱 Testing on {device.upper()} device:")

        conv_warnings = validate_device_compatibility("Conv2D", conv_params, device)
        dense_warnings = validate_device_compatibility("Dense", dense_params, device)

        all_warnings = conv_warnings + dense_warnings

        if all_warnings:
            print(f"   ⚠️  Found {len(all_warnings)} compatibility warnings:")
            for warning in all_warnings:
                print(f"      • {warning}")
        else:
            print(f"   ✅ All layers compatible with {device} device")


def demo_grammar_integration():
    """Demonstrate how this integrates with ANTLR grammar."""
    print("\n🔍 Demo 4: Grammar Integration Concept")
    print("=" * 50)

    print("In your ANTLR grammar, you would have rules like:")
    print(
        """
    // Type-constrained tokens
    POSITIVE_INT    : [1-9] [0-9]* ;                    // Positive integers only
    KERNEL_SIZE_INT : [1-9] | '1' [0-5] ;              // 1-15 for kernel sizes
    STRIDE_INT      : [1-8] ;                           // 1-8 for strides  
    DROPOUT_RATE    : '0.' [0-9] | '0.9' ;              // 0.0-0.9 for dropout
    
    // Layer-specific rules
    conv2d_layer: 'Conv2D' '(' conv2d_params ')'
    conv2d_params: conv2d_param (',' conv2d_param)*
    conv2d_param: 'filters' '=' POSITIVE_INT
                | 'kernel_size' '=' kernel_size_value
                | 'strides' '=' stride_value
                | 'activation' '=' activation_type
    
    activation_type: 'relu' | 'sigmoid' | 'tanh' | 'softmax'
    """
    )

    print("\nWhen the parser encounters:")
    print("   Conv2D(filters=32, kernel_size=3, activation=relu)")
    print("\nIt would:")
    print("   1. ✅ Validate 'filters=32' matches POSITIVE_INT")
    print("   2. ✅ Validate 'kernel_size=3' is within 1-15 range")
    print("   3. ✅ Validate 'activation=relu' is in enum")
    print("   4. ✅ Create type-safe AST node with validated parameters")

    print("\nBut if it encounters:")
    print("   Conv2D(filters=0, kernel_size=20, activation=invalid)")
    print("\nIt would:")
    print("   1. ❌ Reject 'filters=0' (doesn't match POSITIVE_INT)")
    print("   2. ❌ Reject 'kernel_size=20' (exceeds 1-15 range)")
    print("   3. ❌ Reject 'activation=invalid' (not in enum)")
    print("   4. 🛑 Stop parsing with clear error messages")


def run_all_demos():
    """Run all grammar validation demonstrations."""
    print("🚀 Grammar-Based Type Constraints & Validation Demo")
    print("=" * 70)

    try:
        demo_valid_parameters()
        demo_invalid_parameters()
        demo_device_compatibility()
        demo_grammar_integration()

        print(f"\n🎯 Key Benefits Demonstrated:")
        print(f"  ✅ Early error detection at grammar/parse level")
        print(f"  ✅ Type-safe parameter validation with clear constraints")
        print(f"  ✅ Enumerated type validation (activation, padding)")
        print(f"  ✅ Range-constrained numeric values")
        print(f"  ✅ Device-specific compatibility checking")
        print(f"  ✅ Clear, actionable error messages")
        print(f"  ✅ Prevention of invalid DSL code from reaching semantic analysis")

        print(f"\n📚 Integration Points:")
        print(f"  1. ANTLR grammar rules enforce basic type constraints")
        print(f"  2. AST node constructors validate parameter ranges")
        print(f"  3. Device compatibility checked during validation")
        print(f"  4. Semantic analyzer handles higher-level constraints")

    except Exception as e:
        print(f"💥 Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_demos()
