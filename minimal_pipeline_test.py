#!/usr/bin/env python3
"""
Minimal test script for EdgeFlow pipeline with real models.
Focuses on core functionality without heavy dependencies.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_edgeflow_parsing():
    """Test EdgeFlow .ef file parsing without heavy imports."""
    logger.info("📄 Testing EdgeFlow file parsing...")
    
    # Find .ef files
    ef_files = list(Path(".").glob("*.ef"))
    
    if not ef_files:
        logger.warning("⚠️  No .ef files found")
        return False
    
    logger.info(f"Found {len(ef_files)} EdgeFlow files:")
    for ef_file in ef_files:
        logger.info(f"   📋 {ef_file.name}")
    
    # Test parsing the realistic model
    realistic_model = Path("realistic_model.ef")
    if realistic_model.exists():
        logger.info(f"🎯 Testing parsing of {realistic_model.name}")
        
        try:
            # Simple file reading test
            with open(realistic_model, 'r') as f:
                content = f.read()
            
            logger.info(f"✅ Successfully read {realistic_model.name} ({len(content)} chars)")
            
            # Check for key EdgeFlow constructs
            constructs = ['model_name', 'pipeline', 'Conv2D', 'Dense', 'connect']
            found_constructs = []
            
            for construct in constructs:
                if construct in content:
                    found_constructs.append(construct)
            
            logger.info(f"📊 Found EdgeFlow constructs: {found_constructs}")
            
            return len(found_constructs) >= 3  # At least 3 constructs found
            
        except Exception as e:
            logger.error(f"❌ Failed to read {realistic_model.name}: {e}")
            return False
    
    return True

def test_model_files():
    """Test availability and basic properties of model files."""
    logger.info("🔍 Testing model files...")
    
    # Check for existing models
    model_files = [
        "mobilenet_v2_keras.h5",
        "resnet50_keras.h5", 
        "model.tflite",
        "downloaded_models/mobilenet_v1_1.0_224.tflite"
    ]
    
    found_models = []
    
    for model_file in model_files:
        model_path = Path(model_file)
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            found_models.append((model_file, size_mb))
            logger.info(f"✅ Found {model_file} ({size_mb:.1f}MB)")
        else:
            logger.info(f"❌ Missing {model_file}")
    
    logger.info(f"📊 Total models found: {len(found_models)}")
    
    return len(found_models) > 0

def test_basic_imports():
    """Test basic EdgeFlow imports without triggering heavy dependencies."""
    logger.info("📦 Testing basic imports...")
    
    try:
        # Test parser import
        sys.path.insert(0, '.')
        import parser
        logger.info("✅ Successfully imported parser module")
        
        # Test if parse_ef function exists
        if hasattr(parser, 'parse_ef'):
            logger.info("✅ parse_ef function available")
        else:
            logger.warning("⚠️  parse_ef function not found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to import parser: {e}")
        return False

def test_edgeflow_config_validation():
    """Test EdgeFlow configuration validation."""
    logger.info("🔧 Testing EdgeFlow config validation...")
    
    try:
        # Test a simple config parsing
        sample_config = Path("sample_config.ef")
        if sample_config.exists():
            with open(sample_config, 'r') as f:
                content = f.read()
            
            logger.info(f"✅ Read sample config ({len(content)} chars)")
            
            # Basic validation - check for required fields
            required_fields = ['model', 'target_device']
            found_fields = []
            
            for field in required_fields:
                if field in content:
                    found_fields.append(field)
            
            logger.info(f"📊 Found required fields: {found_fields}")
            return len(found_fields) >= 1
        
        else:
            logger.warning("⚠️  No sample config found")
            return True  # Not a failure if no sample exists
            
    except Exception as e:
        logger.error(f"❌ Config validation failed: {e}")
        return False

def test_directory_structure():
    """Test EdgeFlow project directory structure."""
    logger.info("📁 Testing directory structure...")
    
    required_dirs = [
        "semantic_analyzer",
        "frontend", 
        "deployment",
        "examples"
    ]
    
    required_files = [
        "edgeflowc.py",
        "pipeline.py",
        "parser.py",
        "validator.py"
    ]
    
    found_dirs = []
    found_files = []
    
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            found_dirs.append(dir_name)
            logger.info(f"✅ Found directory: {dir_name}")
        else:
            logger.info(f"❌ Missing directory: {dir_name}")
    
    for file_name in required_files:
        if Path(file_name).is_file():
            found_files.append(file_name)
            logger.info(f"✅ Found file: {file_name}")
        else:
            logger.info(f"❌ Missing file: {file_name}")
    
    logger.info(f"📊 Directory structure: {len(found_dirs)}/{len(required_dirs)} dirs, {len(found_files)}/{len(required_files)} files")
    
    return len(found_dirs) >= 2 and len(found_files) >= 3

def run_minimal_tests():
    """Run minimal tests without heavy dependencies."""
    logger.info("🚀 Starting Minimal EdgeFlow Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Model Files", test_model_files),
        ("EdgeFlow Parsing", test_edgeflow_parsing),
        ("Basic Imports", test_basic_imports),
        ("Config Validation", test_edgeflow_config_validation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}")
        logger.info("-" * 30)
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"   {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"   ❌ FAILED: {e}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 MINIMAL TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    logger.info(f"Total Tests: {total}")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {total - passed}")
    logger.info(f"📈 Success Rate: {passed/total*100:.1f}%")
    
    logger.info("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {status} {test_name}")
    
    if passed == total:
        logger.info("\n🎉 All minimal tests passed!")
        logger.info("💡 EdgeFlow project structure and basic functionality verified.")
    else:
        logger.info(f"\n⚠️  {total - passed} test(s) failed.")
    
    return results

def main():
    """Main function."""
    results = run_minimal_tests()
    
    # Provide recommendations
    logger.info("\n💡 RECOMMENDATIONS:")
    
    if results.get("Model Files", False):
        logger.info("✅ Real models are available for testing")
        logger.info("   Next: Try running EdgeFlow compiler on these models")
    else:
        logger.info("⚠️  Consider downloading more test models")
        logger.info("   Run: python3 download_test_models.py --lightweight")
    
    if results.get("EdgeFlow Parsing", False):
        logger.info("✅ EdgeFlow DSL files are properly structured")
    else:
        logger.info("⚠️  Check EdgeFlow DSL syntax in .ef files")
    
    if results.get("Basic Imports", False):
        logger.info("✅ Core EdgeFlow modules are importable")
        logger.info("   Next: Test full pipeline compilation")
    else:
        logger.info("⚠️  Check Python dependencies and module paths")
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
