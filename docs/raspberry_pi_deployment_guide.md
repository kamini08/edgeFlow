# Running EdgeFlow Optimized Models on Raspberry Pi

## Overview

This guide shows you how to deploy and run your EdgeFlow-optimized .tflite models on Raspberry Pi using the built-in deployment infrastructure.

## 🎯 Quick Start

### Option 1: Use Pre-built Deployment Package

```bash
# Extract the deployment package
cd /home/kamini08/projects/edgeFlow
tar -xzf deployment/edgeflow_raspberry_pi_deployment.tar.gz

# Copy to Raspberry Pi
scp -r edgeflow_raspberry_pi_deployment/ pi@your-pi-ip:/home/pi/

# On Raspberry Pi, run deployment
cd /home/pi/edgeflow_raspberry_pi_deployment
sudo ./scripts/deploy.sh model/optimized_mobilenet_v2_keras.h5
```

### Option 2: Manual Setup (Recommended for Development)

## 📋 Prerequisites

### On Raspberry Pi:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip
sudo apt install python3 python3-pip python3-dev -y

# Install TensorFlow Lite runtime
pip3 install tflite-runtime numpy

# Optional: Install OpenCV for image processing
sudo apt install python3-opencv -y
```

## 🚀 Step-by-Step Deployment

### Step 1: Transfer Your Model

```bash
# From your development machine, copy the optimized model
scp model_optimized.tflite pi@your-pi-ip:/home/pi/models/
# OR
scp generated/inference.py pi@your-pi-ip:/home/pi/edgeflow/
```

### Step 2: Copy EdgeFlow Inference Engine

```bash
# Copy the inference engine
scp deployment/edgeflow_raspberry_pi_deployment/inference_code/inference.py pi@your-pi-ip:/home/pi/edgeflow/
```

### Step 3: Create a Simple Test Script

Create this file on your Raspberry Pi as `/home/pi/edgeflow/test_model.py`:

```python
#!/usr/bin/env python3
"""
Test script for EdgeFlow optimized model on Raspberry Pi
"""

import numpy as np
import time
from inference import EdgeFlowInference

def test_model():
    # Initialize with your optimized model
    model_path = "/home/pi/models/model_optimized.tflite"
    
    print("🚀 Loading EdgeFlow optimized model...")
    inference = EdgeFlowInference(model_path)
    
    print("📊 Running benchmark...")
    results = inference.benchmark(num_runs=50)
    
    print(f"✅ Benchmark Results:")
    print(f"   Mean inference time: {results['mean_time_ms']:.2f}ms")
    print(f"   Throughput: {results['throughput_fps']:.2f} FPS")
    print(f"   Min time: {results['min_time_ms']:.2f}ms")
    print(f"   Max time: {results['max_time_ms']:.2f}ms")

if __name__ == "__main__":
    test_model()
```

### Step 4: Run the Model

```bash
# On Raspberry Pi
cd /home/pi/edgeflow
python3 test_model.py
```

## 🔧 Advanced Usage

### Real-time Image Classification

Create `/home/pi/edgeflow/image_classifier.py`:

```python
#!/usr/bin/env python3
"""
Real-time image classification using EdgeFlow optimized model
"""

import cv2
import numpy as np
import time
from inference import EdgeFlowInference

class ImageClassifier:
    def __init__(self, model_path, labels_path=None):
        self.inference = EdgeFlowInference(model_path)
        self.labels = self._load_labels(labels_path) if labels_path else None
        
    def _load_labels(self, labels_path):
        with open(labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def preprocess_image(self, image):
        # Resize to model input size (typically 224x224 for MobileNet)
        input_shape = self.inference.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize and normalize
        image = cv2.resize(image, (width, height))
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def classify_image(self, image):
        # Preprocess
        input_data = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        predictions = self.inference.predict(input_data)
        inference_time = time.time() - start_time
        
        # Get top prediction
        top_index = np.argmax(predictions[0])
        confidence = predictions[0][top_index]
        
        result = {
            'class_index': top_index,
            'confidence': float(confidence),
            'inference_time_ms': inference_time * 1000
        }
        
        if self.labels:
            result['class_name'] = self.labels[top_index]
            
        return result

def main():
    # Initialize classifier
    classifier = ImageClassifier("/home/pi/models/model_optimized.tflite")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("🎥 Starting real-time classification...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Classify frame
        result = classifier.classify_image(frame)
        
        # Display results
        text = f"Class: {result['class_index']} ({result['confidence']:.2f})"
        text += f" | {result['inference_time_ms']:.1f}ms"
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        cv2.imshow('EdgeFlow Classification', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### Batch Processing

Create `/home/pi/edgeflow/batch_processor.py`:

```python
#!/usr/bin/env python3
"""
Batch image processing with EdgeFlow optimized model
"""

import os
import glob
import numpy as np
import cv2
from inference import EdgeFlowInference

def process_images_batch(model_path, input_dir, output_dir):
    # Initialize inference engine
    inference = EdgeFlowInference(model_path)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"📁 Found {len(image_files)} images to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_time = 0
    
    for i, image_path in enumerate(image_files):
        print(f"🖼️  Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        input_shape = inference.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        processed_image = cv2.resize(image, (width, height))
        processed_image = processed_image.astype(np.float32) / 255.0
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Run inference
        start_time = time.time()
        predictions = inference.predict(processed_image)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Save results
        top_class = np.argmax(predictions[0])
        confidence = predictions[0][top_class]
        
        result = {
            'filename': os.path.basename(image_path),
            'class': int(top_class),
            'confidence': float(confidence),
            'inference_time_ms': inference_time * 1000
        }
        results.append(result)
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
        text = f"Class: {top_class} ({confidence:.2f})"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(output_path, image)
    
    # Print summary
    avg_time = total_time / len(image_files) * 1000
    print(f"\n📊 Batch Processing Complete:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Average inference time: {avg_time:.2f}ms")
    print(f"   Total processing time: {total_time:.2f}s")
    print(f"   Throughput: {len(image_files)/total_time:.2f} images/sec")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--input", required=True, help="Input directory with images")
    parser.add_argument("--output", required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    process_images_batch(args.model, args.input, args.output)
```

## 🔧 Performance Optimization

### 1. Enable GPU Acceleration (if available)

```python
# Modify inference.py to use GPU delegate
import tflite_runtime.interpreter as tflite

# Load with GPU delegate
try:
    delegate = tflite.load_delegate('libedgetpu.so.1')
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[delegate]
    )
except:
    # Fallback to CPU
    interpreter = tflite.Interpreter(model_path=model_path)
```

### 2. Optimize System Settings

```bash
# Increase GPU memory split
sudo raspi-config
# Advanced Options -> Memory Split -> 128

# Increase swap file
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 3. Monitor Performance

Create `/home/pi/edgeflow/monitor.py`:

```python
#!/usr/bin/env python3
"""
System monitoring during inference
"""

import psutil
import time
import threading
from inference import EdgeFlowInference

class SystemMonitor:
    def __init__(self):
        self.monitoring = False
        self.stats = []
    
    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        self.monitor_thread.join()
    
    def _monitor_loop(self):
        while self.monitoring:
            stats = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'temperature': self._get_cpu_temperature(),
                'timestamp': time.time()
            }
            self.stats.append(stats)
            time.sleep(0.1)
    
    def _get_cpu_temperature(self):
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
            return temp
        except:
            return None
    
    def get_summary(self):
        if not self.stats:
            return {}
        
        cpu_usage = [s['cpu_percent'] for s in self.stats]
        memory_usage = [s['memory_percent'] for s in self.stats]
        temperatures = [s['temperature'] for s in self.stats if s['temperature']]
        
        return {
            'avg_cpu_percent': sum(cpu_usage) / len(cpu_usage),
            'max_cpu_percent': max(cpu_usage),
            'avg_memory_percent': sum(memory_usage) / len(memory_usage),
            'max_memory_percent': max(memory_usage),
            'avg_temperature': sum(temperatures) / len(temperatures) if temperatures else None,
            'max_temperature': max(temperatures) if temperatures else None
        }

def benchmark_with_monitoring(model_path):
    # Initialize
    inference = EdgeFlowInference(model_path)
    monitor = SystemMonitor()
    
    print("🔍 Starting monitored benchmark...")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Run benchmark
    results = inference.benchmark(num_runs=100)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Get system stats
    system_stats = monitor.get_summary()
    
    print(f"\n📊 Performance Results:")
    print(f"   Inference time: {results['mean_time_ms']:.2f}ms")
    print(f"   Throughput: {results['throughput_fps']:.2f} FPS")
    print(f"   CPU usage: {system_stats['avg_cpu_percent']:.1f}% (max: {system_stats['max_cpu_percent']:.1f}%)")
    print(f"   Memory usage: {system_stats['avg_memory_percent']:.1f}% (max: {system_stats['max_memory_percent']:.1f}%)")
    if system_stats['avg_temperature']:
        print(f"   Temperature: {system_stats['avg_temperature']:.1f}°C (max: {system_stats['max_temperature']:.1f}°C)")

if __name__ == "__main__":
    benchmark_with_monitoring("/home/pi/models/model_optimized.tflite")
```

## 🚀 Running as a Service

### Create systemd service:

```bash
sudo nano /etc/systemd/system/edgeflow-inference.service
```

```ini
[Unit]
Description=EdgeFlow ML Inference Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/edgeflow
ExecStart=/usr/bin/python3 /home/pi/edgeflow/image_classifier.py
Restart=always
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable edgeflow-inference.service
sudo systemctl start edgeflow-inference.service

# Check status
sudo systemctl status edgeflow-inference.service
```

## 📊 Expected Performance

### Raspberry Pi 4 (4GB RAM):
- **MobileNet V2 (optimized)**: ~15-25ms inference time
- **MobileNet V1 (optimized)**: ~10-20ms inference time
- **Small custom models**: ~5-15ms inference time

### Raspberry Pi 3B+:
- **MobileNet V2 (optimized)**: ~25-40ms inference time
- **MobileNet V1 (optimized)**: ~20-35ms inference time

## 🔧 Troubleshooting

### Common Issues:

1. **TensorFlow Lite not found**:
   ```bash
   pip3 install tflite-runtime
   ```

2. **Memory issues**:
   ```bash
   # Increase swap
   sudo dphys-swapfile swapoff
   sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

3. **Camera not working**:
   ```bash
   # Enable camera
   sudo raspi-config
   # Interface Options -> Camera -> Enable
   ```

4. **Slow inference**:
   - Use INT8 quantized models
   - Reduce input image size
   - Enable GPU memory split
   - Close unnecessary processes

## 🎉 Next Steps

1. **Deploy your optimized model**: Use the scripts above with your specific .tflite file
2. **Monitor performance**: Use the monitoring tools to optimize further
3. **Scale deployment**: Use the systemd service for production deployment
4. **Add custom preprocessing**: Modify the image preprocessing for your specific use case

Your EdgeFlow-optimized models are now ready to run efficiently on Raspberry Pi! 🚀
