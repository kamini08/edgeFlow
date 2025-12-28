import os
import sys
import platform
import tensorflow as tf

def check_tensorflow():
    print("=== System Information ===")
    print(f"Python: {sys.version}")
    print(f"OS: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    print(f"CPU: {platform.processor()}")
    print(f"Python Implementation: {platform.python_implementation()}")
    
    print("\n=== TensorFlow Information ===")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    
    print("\n=== GPU Information ===")
    print(f"CUDA Available: {tf.test.is_built_with_cuda()}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    print("\n=== Basic TensorFlow Operations ===")
    try:
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = a + b
        print(f"Tensor addition test: {c.numpy()}")
        print("✅ Basic TensorFlow operations working")
    except Exception as e:
        print(f"❌ Error in TensorFlow operations: {str(e)}")

if __name__ == "__main__":
    print("Running minimal TensorFlow test...")
    check_tensorflow()
    print("\nTest completed.")
