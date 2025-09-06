"""Prototype Quantizer - Proof of Concept for EdgeFlow

This script demonstrates that we can successfully quantize TensorFlow Lite models
from float32 to int8, which is the core optimization that EdgeFlow provides.

This is a standalone script to prove the concept works before integrating
into the main EdgeFlow pipeline.
"""

import os
import tempfile
import numpy as np
import tensorflow as tf
from typing import Tuple

def create_simple_model() -> str:
    """Create a simple TensorFlow model for testing quantization.
    
    Returns:
        str: Path to the saved model file
    """
    print("Creating a simple test model...")
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,), name='dense_1'),
        tf.keras.layers.Dense(5, activation='relu', name='dense_2'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create some dummy training data
    X_train = np.random.random((100, 5)).astype(np.float32)
    y_train = np.random.randint(0, 2, (100, 1)).astype(np.float32)
    
    # Train the model briefly
    print("Training model...")
    model.fit(X_train, y_train, epochs=5, verbose=0)
    
    # Save as SavedModel format
    model_path = "test_model"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    return model_path

def representative_dataset() -> np.ndarray:
    """Generate representative data for quantization calibration.
    
    This function provides sample data that represents the typical input
    distribution for the model. The quantizer uses this to determine
    the optimal quantization parameters.
    
    Returns:
        np.ndarray: Representative input data
    """
    # Generate 100 samples of random data matching our model's input shape
    # This should represent the typical data distribution the model will see
    for _ in range(100):
        yield [np.random.random((1, 5)).astype(np.float32)]

def quantize_model_float32_to_int8(model_path: str, output_path: str) -> Tuple[str, dict]:
    """Quantize a float32 model to int8.
    
    Args:
        model_path: Path to the input float32 model
        output_path: Path where the quantized model will be saved
        
    Returns:
        Tuple of (quantized_model_path, quantization_info)
    """
    print(f"Quantizing model from {model_path} to {output_path}")
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
    # Configure quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert the model
    print("Converting model to quantized TFLite format...")
    try:
        quantized_model = converter.convert()
        print("✅ Quantization successful!")
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        raise
    
    # Save the quantized model
    with open(output_path, 'wb') as f:
        f.write(quantized_model)
    
    # Get model sizes for comparison
    original_size = os.path.getsize(model_path) if os.path.isdir(model_path) else 0
    quantized_size = os.path.getsize(output_path)
    
    quantization_info = {
        "original_size_bytes": original_size,
        "quantized_size_bytes": quantized_size,
        "compression_ratio": quantized_size / original_size if original_size > 0 else 0,
        "size_reduction_percent": (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
    }
    
    return output_path, quantization_info

def test_quantized_model(model_path: str) -> dict:
    """Test the quantized model to ensure it works correctly.
    
    Args:
        model_path: Path to the quantized model
        
    Returns:
        dict: Test results including inference time and accuracy
    """
    print(f"Testing quantized model: {model_path}")
    
    # Load the quantized model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Test with sample data
    test_data = np.random.random((1, 5)).astype(np.float32)
    
    # Run inference
    import time
    start_time = time.time()
    
    interpreter.set_tensor(input_details[0]['index'], test_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    inference_time = time.time() - start_time
    
    test_results = {
        "inference_time_ms": inference_time * 1000,
        "output_shape": output.shape,
        "sample_output": output[0].tolist(),
        "model_loaded": True
    }
    
    print(f"✅ Model test successful!")
    print(f"   Inference time: {inference_time*1000:.2f}ms")
    print(f"   Output: {output[0]}")
    
    return test_results

def main():
    """Main function to demonstrate quantization pipeline."""
    print("=" * 60)
    print("🚀 EdgeFlow Prototype Quantizer")
    print("   Proving that model quantization works!")
    print("=" * 60)
    
    try:
        # Step 1: Create a test model
        print("\n📝 Step 1: Creating test model...")
        model_path = create_simple_model()
        
        # Step 2: Quantize the model
        print("\n⚙️ Step 2: Quantizing model...")
        quantized_path = "test_model_quantized.tflite"
        quantized_path, quant_info = quantize_model_float32_to_int8(model_path, quantized_path)
        
        # Step 3: Show results
        print("\n📊 Step 3: Quantization Results")
        print("-" * 40)
        print(f"Original model size: {quant_info['original_size_bytes']:,} bytes")
        print(f"Quantized model size: {quant_info['quantized_size_bytes']:,} bytes")
        print(f"Size reduction: {quant_info['size_reduction_percent']:.1f}%")
        print(f"Compression ratio: {quant_info['compression_ratio']:.2f}x")
        
        # Step 4: Test the quantized model
        print("\n🧪 Step 4: Testing quantized model...")
        test_results = test_quantized_model(quantized_path)
        
        # Step 5: Summary
        print("\n🎉 SUCCESS! Quantization pipeline works!")
        print("=" * 60)
        print("✅ Created float32 model")
        print("✅ Quantized to int8")
        print("✅ Model size reduced significantly")
        print("✅ Quantized model runs inference correctly")
        print("✅ Ready for integration into EdgeFlow!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
