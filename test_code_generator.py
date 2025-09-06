#!/usr/bin/env python3
"""Test script for EdgeFlow Code Generator."""

from edgeflow_ast import create_program_from_dict
from parser import parse_ef
from code_generator import CodeGenerator, generate_code


def test_code_generator():
    """Test the code generator with a sample configuration."""
    print("=== Testing EdgeFlow Code Generator ===")
    
    # Create a test configuration file
    test_config = """model: "mobilenet_v2.tflite"
quantize: int8
target_device: raspberry_pi
deploy_path: "/models/"
input_stream: camera
buffer_size: 32
optimize_for: latency
memory_limit: 64
enable_fusion: true
"""
    
    with open('test_config.ef', 'w') as f:
        f.write(test_config)
    
    try:
        # Parse configuration
        print("1. Parsing configuration...")
        config = parse_ef('test_config.ef')
        print(f"   Parsed config: {len(config)} keys")
        
        # Create AST
        print("2. Creating AST...")
        program = create_program_from_dict(config)
        print(f"   AST created with {len(program.statements)} statements")
        
        # Generate code
        print("3. Generating code...")
        generator = CodeGenerator(program)
        
        # Generate Python code
        python_code = generator.generate_python_inference()
        print(f"   Python code generated: {len(python_code)} characters")
        
        # Generate C++ code
        cpp_code = generator.generate_cpp_inference()
        print(f"   C++ code generated: {len(cpp_code)} characters")
        
        # Generate optimization report
        report = generator.generate_optimization_report()
        print(f"   Optimization report generated: {len(report)} characters")
        
        # Save generated files
        print("4. Saving generated files...")
        files = generate_code(program, 'generated')
        for file_type, file_path in files.items():
            print(f"   {file_type}: {file_path}")
        
        # Show sample of generated Python code
        print("\n=== Sample Generated Python Code ===")
        lines = python_code.split('\n')
        for i, line in enumerate(lines[:30]):  # Show first 30 lines
            print(f"{i+1:3d}: {line}")
        if len(lines) > 30:
            print(f"... ({len(lines) - 30} more lines)")
        
        # Show optimization report
        print("\n=== Optimization Report ===")
        print(report)
        
        print("\n✅ Code generation test completed successfully!")
        
    finally:
        # Clean up test file
        import os
        if os.path.exists('test_config.ef'):
            os.remove('test_config.ef')


if __name__ == "__main__":
    test_code_generator()

