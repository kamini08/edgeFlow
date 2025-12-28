#!/usr/bin/env python3
"""
Quick deployment script for EdgeFlow models to Raspberry Pi
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run shell command and return result."""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Command failed: {result.stderr}")
        sys.exit(1)
    return result


def deploy_to_raspberry_pi(pi_ip, model_path, pi_user="pi"):
    """Deploy EdgeFlow model to Raspberry Pi."""

    print("🚀 EdgeFlow Raspberry Pi Deployment")
    print("=" * 40)

    # Validate inputs
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    print(f"📱 Target Pi: {pi_user}@{pi_ip}")
    print(f"🤖 Model: {model_path}")

    # Step 1: Create directories on Pi
    print("\n📁 Step 1: Setting up directories on Pi...")
    run_command(f"ssh {pi_user}@{pi_ip} 'mkdir -p /home/{pi_user}/edgeflow/models'")

    # Step 2: Copy model file
    print("\n📤 Step 2: Copying model to Pi...")
    model_name = Path(model_path).name
    run_command(f"scp {model_path} {pi_user}@{pi_ip}:/home/{pi_user}/edgeflow/models/")

    # Step 3: Copy inference engine
    print("\n🔧 Step 3: Copying inference engine...")
    inference_script = (
        "deployment/edgeflow_raspberry_pi_deployment/inference_code/inference.py"
    )
    if Path(inference_script).exists():
        run_command(
            f"scp {inference_script} {pi_user}@{pi_ip}:/home/{pi_user}/edgeflow/"
        )
    else:
        print(f"⚠️  Inference script not found at {inference_script}")
        print("   Using generated inference code instead...")
        if Path("generated/inference.py").exists():
            run_command(
                f"scp generated/inference.py {pi_user}@{pi_ip}:/home/{pi_user}/edgeflow/"
            )
        else:
            print("❌ No inference script found!")
            sys.exit(1)

    # Step 4: Install dependencies on Pi
    print("\n📦 Step 4: Installing dependencies on Pi...")
    install_cmd = f"""
    ssh {pi_user}@{pi_ip} '
    sudo apt update && 
    sudo apt install -y python3-pip python3-dev && 
    pip3 install tflite-runtime numpy opencv-python-headless
    '
    """
    run_command(install_cmd)

    # Step 5: Create test script
    print("\n📝 Step 5: Creating test script...")
    test_script = f"""#!/usr/bin/env python3
import sys
sys.path.append('/home/{pi_user}/edgeflow')

from inference import EdgeFlowInference
import numpy as np

def test_model():
    model_path = "/home/{pi_user}/edgeflow/models/{model_name}"
    print(f"🤖 Testing model: {{model_path}}")
    
    try:
        inference = EdgeFlowInference(model_path)
        print("✅ Model loaded successfully!")
        
        # Run benchmark
        print("🏃 Running benchmark...")
        results = inference.benchmark(num_runs=10)
        
        print(f"📊 Results:")
        print(f"   Mean time: {{results['mean_time_ms']:.2f}}ms")
        print(f"   Throughput: {{results['throughput_fps']:.2f}} FPS")
        print("🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {{e}}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
"""

    # Write test script to temporary file and copy
    with open("/tmp/test_edgeflow.py", "w") as f:
        f.write(test_script)

    run_command(
        f"scp /tmp/test_edgeflow.py {pi_user}@{pi_ip}:/home/{pi_user}/edgeflow/"
    )
    run_command(
        f"ssh {pi_user}@{pi_ip} 'chmod +x /home/{pi_user}/edgeflow/test_edgeflow.py'"
    )

    # Step 6: Test the deployment
    print("\n🧪 Step 6: Testing deployment...")
    test_result = run_command(
        f"ssh {pi_user}@{pi_ip} 'cd /home/{pi_user}/edgeflow && python3 test_edgeflow.py'",
        check=False,
    )

    if test_result.returncode == 0:
        print("✅ Deployment successful!")
        print(test_result.stdout)
    else:
        print("⚠️  Test failed, but deployment may still work:")
        print(test_result.stderr)

    # Step 7: Provide usage instructions
    print("\n💡 Usage Instructions:")
    print(f"   SSH to Pi: ssh {pi_user}@{pi_ip}")
    print(f"   Navigate: cd /home/{pi_user}/edgeflow")
    print(f"   Test model: python3 test_edgeflow.py")
    print(
        f"   Run inference: python3 inference.py --model models/{model_name} --benchmark"
    )

    print("\n🎉 Deployment completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy EdgeFlow model to Raspberry Pi"
    )
    parser.add_argument("pi_ip", help="Raspberry Pi IP address")
    parser.add_argument("model_path", help="Path to .tflite model file")
    parser.add_argument("--user", default="pi", help="Pi username (default: pi)")

    args = parser.parse_args()

    deploy_to_raspberry_pi(args.pi_ip, args.model_path, args.user)


if __name__ == "__main__":
    main()
