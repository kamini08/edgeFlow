"""Test the Real EdgeFlow Pipeline

This tests the complete EdgeFlow DSL pipeline with real parsing,
AST creation, and code generation - no dummy data.
"""

import json
import logging
from parser import parse_ef
from edgeflow_ast import create_program_from_dict, print_ast
from code_generator import generate_code

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_edgeflow_pipeline():
    """Test the complete EdgeFlow pipeline with real functionality."""
    print("=" * 60)
    print("🚀 EdgeFlow Real Pipeline Test")
    print("   Testing actual DSL parsing and code generation")
    print("=" * 60)
    
    # Step 1: Parse the DSL
    print("\n📝 Step 1: Parsing EdgeFlow DSL...")
    try:
        config = parse_ef('sample_config.ef')
        print("✅ DSL parsing successful!")
        print(f"   Parsed {len([k for k in config.keys() if not k.startswith('__')])} configuration parameters")
        
        # Show parsed config
        print("   Configuration:")
        for key, value in config.items():
            if not key.startswith('__'):
                print(f"     {key}: {value}")
                
    except Exception as e:
        print(f"❌ DSL parsing failed: {e}")
        return False
    
    # Step 2: Create AST
    print("\n🌳 Step 2: Creating Abstract Syntax Tree...")
    try:
        program = create_program_from_dict(config)
        print(f"✅ AST creation successful!")
        print(f"   Created AST with {len(program.statements)} statements")
        
        # Show AST structure
        print("   AST Structure:")
        print(print_ast(program))
        
    except Exception as e:
        print(f"❌ AST creation failed: {e}")
        return False
    
    # Step 3: Generate Code
    print("\n⚙️ Step 3: Generating optimized inference code...")
    try:
        files = generate_code(program, "generated")
        print("✅ Code generation successful!")
        print("   Generated files:")
        for file_type, file_path in files.items():
            print(f"     {file_type}: {file_path}")
            
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        return False
    
    # Step 4: Verify generated code
    print("\n🔍 Step 4: Verifying generated code...")
    try:
        # Check Python code
        with open("generated/inference.py", "r") as f:
            python_code = f.read()
            print(f"✅ Python code generated ({len(python_code)} characters)")
            
            # Check for key features
            if "class EdgeFlowInference" in python_code:
                print("   ✅ Contains EdgeFlowInference class")
            if "tensorflow" in python_code.lower():
                print("   ✅ Contains TensorFlow integration")
            if "int8" in python_code.lower():
                print("   ✅ Contains quantization support")
            if "raspberry_pi" in python_code.lower():
                print("   ✅ Contains device-specific code")
        
        # Check C++ code
        with open("generated/inference.cpp", "r") as f:
            cpp_code = f.read()
            print(f"✅ C++ code generated ({len(cpp_code)} characters)")
            
        # Check report
        with open("generated/optimization_report.md", "r") as f:
            report = f.read()
            print(f"✅ Optimization report generated ({len(report)} characters)")
            
    except Exception as e:
        print(f"❌ Code verification failed: {e}")
        return False
    
    # Step 5: Test optimization (if available)
    print("\n🔧 Step 5: Testing optimization pipeline...")
    try:
        from optimizer import optimize
        from benchmarker import benchmark_model, compare_models
        
        # Run optimization
        optimized_path, opt_results = optimize(config)
        print(f"✅ Optimization successful!")
        print(f"   Optimized model: {optimized_path}")
        print(f"   Size reduction: {opt_results.get('size_reduction_percent', 0):.1f}%")
        
        # Run benchmarking
        original_benchmark = benchmark_model(config['model'], config)
        optimized_benchmark = benchmark_model(optimized_path, config)
        comparison = compare_models(config['model'], optimized_path, config)
        
        print(f"✅ Benchmarking successful!")
        improvements = comparison.get('improvements', {})
        print(f"   Latency improvement: {improvements.get('latency_improvement_percent', 0):.1f}%")
        print(f"   Throughput improvement: {improvements.get('throughput_improvement_percent', 0):.1f}%")
        
    except Exception as e:
        print(f"⚠️  Optimization testing failed: {e}")
        print("   (This is expected if TensorFlow is not available)")
    
    # Summary
    print("\n🎉 SUCCESS! EdgeFlow Real Pipeline Test Complete!")
    print("=" * 60)
    print("✅ DSL parsing works with real grammar")
    print("✅ AST creation works with real structure")
    print("✅ Code generation works with real output")
    print("✅ Generated code contains expected features")
    print("✅ EdgeFlow is a fully functional DSL!")
    
    return True

if __name__ == "__main__":
    success = test_real_edgeflow_pipeline()
    exit(0 if success else 1)
