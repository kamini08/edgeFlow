#!/usr/bin/env python3
"""
Raspberry Pi System Monitor for EdgeFlow Model Performance
Monitors CPU, memory, temperature, and inference performance in real-time
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil


class RaspberryPiMonitor:
    """Real-time system monitoring for Raspberry Pi during model inference."""

    def __init__(self, log_file: str = "pi_performance.log"):
        self.log_file = log_file
        self.monitoring = False
        self.stats: List[Dict[str, Any]] = []
        self.inference_stats: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None

        # System info
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total / (1024**3)  # GB

        print("🔍 Raspberry Pi Monitor Initialized")
        print(f"   CPU cores: {self.cpu_count}")
        print(f"   Total memory: {self.memory_total:.2f}GB")

    def get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature in Celsius."""
        try:
            {
                "original_stats": {
                    "size_mb": 0.001953,
                    "latency_ms": 0.019530000000000002,
                    "model_path": "uploaded_model.tflite",
                },
                "optimized_stats": {
                    "size_mb": 0.0009765,
                    "latency_ms": 0.007812,
                    "model_path": "optimized_model.tflite",
                },
                "improvements": {
                    "size_reduction": 50.0,
                    "speedup": 2.5,
                    "latency_reduction": 60.0,
                    "throughput_increase": 150.0,
                    "memory_saved": 0.0,
                },
                "config": {"quantize": "int8", "target_device": "raspberry_pi"},
                "timestamp": "2025-09-24T12:51:00.830617",
            }  # Try vcgencmd first (Raspberry Pi specific)
            result = subprocess.run(
                ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                # Extract temperature from "temp=XX.X'C"
                temp = float(temp_str.split("=")[1].split("'")[0])
                return temp
        except Exception:
            pass

        try:
            # Fallback to thermal zone
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read().strip()) / 1000.0
            return temp
        except (FileNotFoundError, PermissionError, ValueError) as e:
            print(f"⚠️  Could not read CPU temperature: {e}")
            return None
        except Exception:
            return None

    def get_gpu_memory(self) -> Dict[str, Optional[int]]:
        """Get GPU memory usage (Raspberry Pi specific)."""
        try:
            # Get GPU memory split
            result = subprocess.run(
                ["vcgencmd", "get_mem", "gpu"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            gpu_mem = None
            if result.returncode == 0:
                gpu_mem = int(result.stdout.strip().split("=")[1].replace("M", ""))

            # Get GPU memory usage
            result = subprocess.run(
                ["vcgencmd", "get_mem", "reloc"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            gpu_used = None
            if result.returncode == 0:
                gpu_used = int(result.stdout.strip().split("=")[1].replace("M", ""))

            return {"total": gpu_mem, "used": gpu_used}
        except Exception:
            return {"total": None, "used": None}

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

        # Memory usage
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk I/O
        disk_io = psutil.disk_io_counters()

        # Network I/O
        net_io = psutil.net_io_counters()

        # Temperature
        temperature = self.get_cpu_temperature()

        # GPU memory
        gpu_memory = self.get_gpu_memory()

        # Load average
        load_avg = os.getloadavg()

        return {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "cpu": {
                "percent": cpu_percent,
                "per_core": cpu_per_core,
                "load_avg_1m": load_avg[0],
                "load_avg_5m": load_avg[1],
                "load_avg_15m": load_avg[2],
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "used_gb": memory.used / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent": memory.percent,
                "swap_used_gb": swap.used / (1024**3),
                "swap_percent": swap.percent,
            },
            "temperature": {"cpu_celsius": temperature},
            "gpu": gpu_memory,
            "disk_io": {
                "read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
                "write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0,
            },
            "network_io": {
                "bytes_sent_mb": net_io.bytes_sent / (1024**2) if net_io else 0,
                "bytes_recv_mb": net_io.bytes_recv / (1024**2) if net_io else 0,
            },
        }

    def start_monitoring(self, interval: float = 0.5):
        """Start system monitoring in background thread."""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"📊 System monitoring started (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=2)
        print("⏹️  System monitoring stopped")

    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                stats = self.get_system_stats()
                self.stats.append(stats)

                # Keep only last 1000 samples to prevent memory issues
                if len(self.stats) > 1000:
                    self.stats = self.stats[-1000:]

                time.sleep(interval)
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(interval)

    def log_inference_stats(
        self, inference_time_ms: float, model_name: str = "unknown"
    ):
        """Log inference performance statistics."""
        inference_stat = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "model_name": model_name,
            "inference_time_ms": inference_time_ms,
            "fps": 1000.0 / inference_time_ms if inference_time_ms > 0 else 0,
        }

        # Add current system stats
        if self.stats:
            current_stats = self.stats[-1]
            inference_stat.update(
                {
                    "cpu_percent": current_stats["cpu"]["percent"],
                    "memory_percent": current_stats["memory"]["percent"],
                    "temperature": current_stats["temperature"]["cpu_celsius"],
                }
            )

        self.inference_stats.append(inference_stat)

        # Keep only last 1000 inference stats
        if len(self.inference_stats) > 1000:
            self.inference_stats = self.inference_stats[-1000:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.stats:
            return {"error": "No monitoring data available"}

        # System stats
        cpu_usage = [s["cpu"]["percent"] for s in self.stats]
        memory_usage = [s["memory"]["percent"] for s in self.stats]
        temperatures = [
            s["temperature"]["cpu_celsius"]
            for s in self.stats
            if s["temperature"]["cpu_celsius"] is not None
        ]
        load_avg = [s["cpu"]["load_avg_1m"] for s in self.stats]

        # Inference stats
        inference_times = [s["inference_time_ms"] for s in self.inference_stats]
        fps_values = [s["fps"] for s in self.inference_stats]

        summary: Dict[str, Any] = {
            "monitoring_duration_s": (
                time.time() - self.start_time if self.start_time is not None else 0
            ),
            "total_samples": len(self.stats),
            "total_inferences": len(self.inference_stats),
            "system_performance": {
                "cpu": {
                    "avg_percent": np.mean(cpu_usage),
                    "max_percent": np.max(cpu_usage),
                    "min_percent": np.min(cpu_usage),
                    "std_percent": np.std(cpu_usage),
                },
                "memory": {
                    "avg_percent": np.mean(memory_usage),
                    "max_percent": np.max(memory_usage),
                    "min_percent": np.min(memory_usage),
                },
                "load_average": {
                    "avg": np.mean(load_avg),
                    "max": np.max(load_avg),
                    "min": np.min(load_avg),
                },
            },
        }

        if temperatures:
            summary["system_performance"]["temperature"] = {
                "avg_celsius": np.mean(temperatures),
                "max_celsius": np.max(temperatures),
                "min_celsius": np.min(temperatures),
            }

        if inference_times:
            summary["inference_performance"] = {
                "avg_time_ms": np.mean(inference_times),
                "min_time_ms": np.min(inference_times),
                "max_time_ms": np.max(inference_times),
                "std_time_ms": np.std(inference_times),
                "avg_fps": np.mean(fps_values),
                "max_fps": np.max(fps_values),
                "min_fps": np.min(fps_values),
            }

        return summary

    def print_real_time_stats(self):
        """Print real-time statistics to console."""
        if not self.stats:
            print("No monitoring data available")
            return

        latest = self.stats[-1]

        print(f"\n🔍 Real-time System Stats ({latest['datetime']})")
        print("=" * 60)
        print(
            f"CPU Usage:     {latest['cpu']['percent']:6.1f}% "
            f"(Load: {latest['cpu']['load_avg_1m']:.2f})"
        )
        print(
            f"Memory Usage:  {latest['memory']['percent']:6.1f}% "
            f"({latest['memory']['used_gb']:.2f}GB/{latest['memory']['total_gb']:.2f}GB)"
        )

        if latest["temperature"]["cpu_celsius"]:
            temp = latest["temperature"]["cpu_celsius"]
            temp_status = "🔥" if temp > 70 else "⚠️" if temp > 60 else "✅"
            print(f"CPU Temp:      {temp:6.1f}°C {temp_status}")

        if latest["gpu"]["total"]:
            print(f"GPU Memory:    {latest['gpu']['total']}MB allocated")

        # Per-core CPU usage
        core_usage = " | ".join(
            [
                f"Core{i}: {usage:4.1f}%"
                for i, usage in enumerate(latest["cpu"]["per_core"])
            ]
        )
        print(f"CPU Cores:     {core_usage}")

        # Recent inference stats
        if self.inference_stats:
            recent_inferences = self.inference_stats[-5:]  # Last 5 inferences
            avg_time = np.mean([s["inference_time_ms"] for s in recent_inferences])
            avg_fps = np.mean([s["fps"] for s in recent_inferences])
            print(f"Recent Inf:    {avg_time:6.1f}ms ({avg_fps:5.1f} FPS)")

    def save_detailed_log(self, filename: Optional[str] = None):
        """Save detailed monitoring data to file."""
        if filename is None:
            filename = f"pi_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            "summary": self.get_performance_summary(),
            "system_stats": self.stats,
            "inference_stats": self.inference_stats,
            "metadata": {
                "cpu_count": self.cpu_count,
                "memory_total_gb": self.memory_total,
                "monitoring_start": self.start_time,
                "monitoring_end": time.time(),
            },
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"📄 Detailed log saved: {filename}")
        return filename


def run_htop_like_monitor():
    """Run htop-like real-time monitoring display."""
    monitor = RaspberryPiMonitor()
    monitor.start_monitoring(interval=1.0)

    try:
        print("🖥️  Real-time System Monitor (Press Ctrl+C to stop)")
        print("=" * 60)

        while True:
            # Clear screen (works on most terminals)
            os.system("clear" if os.name == "posix" else "cls")

            monitor.print_real_time_stats()

            # Show summary if we have enough data
            if len(monitor.stats) > 10:
                summary = monitor.get_performance_summary()
                print("\n📊 Session Summary:")
                print(f"   Monitoring time: {summary['monitoring_duration_s']:.1f}s")
                print(
                    f"   Avg CPU: {summary['system_performance']['cpu']['avg_percent']:.1f}%"
                )
                print(
                    f"   Avg Memory: {summary['system_performance']['memory']['avg_percent']:.1f}%"
                )

                if "temperature" in summary["system_performance"]:
                    print(
                        f"   Avg Temp: "
                        f"{summary['system_performance']['temperature']['avg_celsius']:.1f}°C"
                    )

            print("\n💡 Press Ctrl+C to stop monitoring and save log")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⏹️  Stopping monitor...")
        monitor.stop_monitoring()

        # Save log
        monitor.save_detailed_log()

        # Print final summary
        summary = monitor.get_performance_summary()
        print("\n📊 Final Performance Summary:")
        print("=" * 40)
        print(f"Monitoring Duration: {summary['monitoring_duration_s']:.1f}s")
        print(f"Total Samples: {summary['total_samples']}")

        if "inference_performance" in summary:
            print(f"Total Inferences: {summary['total_inferences']}")
            print(
                f"Avg Inference Time: "
                f"{summary['inference_performance']['avg_time_ms']:.2f}ms"
            )
            print(f"Avg FPS: {summary['inference_performance']['avg_fps']:.2f}")

        print(
            f"Avg CPU Usage: {summary['system_performance']['cpu']['avg_percent']:.1f}%"
        )
        print(
            f"Max CPU Usage: {summary['system_performance']['cpu']['max_percent']:.1f}%"
        )
        print(
            f"Avg Memory Usage: "
            f"{summary['system_performance']['memory']['avg_percent']:.1f}%"
        )

        if "temperature" in summary["system_performance"]:
            print(
                f"Avg Temperature: "
                f"{summary['system_performance']['temperature']['avg_celsius']:.1f}°C"
            )
            print(
                f"Max Temperature: "
                f"{summary['system_performance']['temperature']['max_celsius']:.1f}°C"
            )


def monitor_inference_with_model(model_path: str, num_inferences: int = 100):
    """Monitor system while running model inference."""
    print(f"🤖 Monitoring model inference: {model_path}")

    # Import inference engine
    try:
        sys.path.append("/home/pi/edgeflow")
        from inference import EdgeFlowInference
    except ImportError:
        print("❌ EdgeFlow inference engine not found")
        print("   Make sure you've deployed the model first")
        return

    # Initialize monitoring
    monitor = RaspberryPiMonitor()
    monitor.start_monitoring(interval=0.1)  # High frequency monitoring

    try:
        # Initialize inference engine
        print("🚀 Loading model...")
        inference = EdgeFlowInference(model_path)

        # Get input shape for test data
        input_shape = inference.input_details[0]["shape"]
        input_dtype = inference.input_details[0]["dtype"]

        print(f"📊 Running {num_inferences} inferences...")
        print("   (Monitoring CPU, memory, temperature in real-time)")

        inference_times = []

        for i in range(num_inferences):
            # Generate test input
            if input_dtype == np.float32:
                test_input = np.random.random(input_shape).astype(np.float32)
            elif input_dtype == np.int8:
                test_input = np.random.randint(
                    -128, 127, size=input_shape, dtype=np.int8
                )
            else:
                test_input = np.random.random(input_shape).astype(np.float32)

            # Run inference with timing
            start_time = time.perf_counter()
            inference.predict(test_input)
            inference_time = (time.perf_counter() - start_time) * 1000  # ms

            inference_times.append(inference_time)
            monitor.log_inference_stats(inference_time, Path(model_path).name)

            # Print progress every 10 inferences
            if (i + 1) % 10 == 0:
                recent_avg = np.mean(inference_times[-10:])
                print(
                    f"   Progress: {i+1}/{num_inferences} | "
                    f"Recent avg: {recent_avg:.2f}ms"
                )

        print("✅ Inference testing completed!")

    except Exception as e:
        print(f"❌ Inference monitoring failed: {e}")

    finally:
        monitor.stop_monitoring()

        # Save detailed log
        log_file = monitor.save_detailed_log()

        # Print comprehensive summary
        summary = monitor.get_performance_summary()
        print("\n📊 Comprehensive Performance Report:")
        print("=" * 50)

        if "inference_performance" in summary:
            inf_perf = summary["inference_performance"]
            print("🤖 Inference Performance:")
            print(f"   Total inferences: {summary['total_inferences']}")
            print(f"   Avg time: {inf_perf['avg_time_ms']:.2f}ms")
            print(f"   Min time: {inf_perf['min_time_ms']:.2f}ms")
            print(f"   Max time: {inf_perf['max_time_ms']:.2f}ms")
            print(f"   Std dev: {inf_perf['std_time_ms']:.2f}ms")
            print(f"   Avg FPS: {inf_perf['avg_fps']:.2f}")
            print(f"   Max FPS: {inf_perf['max_fps']:.2f}")

        sys_perf = summary["system_performance"]
        print("\n💻 System Performance:")
        print(f"   Avg CPU: {sys_perf['cpu']['avg_percent']:.1f}%")
        print(f"   Max CPU: {sys_perf['cpu']['max_percent']:.1f}%")
        print(f"   Avg Memory: {sys_perf['memory']['avg_percent']:.1f}%")
        print(f"   Max Memory: {sys_perf['memory']['max_percent']:.1f}%")
        print(f"   Avg Load: {sys_perf['load_average']['avg']:.2f}")

        if "temperature" in sys_perf:
            print(f"   Avg Temp: {sys_perf['temperature']['avg_celsius']:.1f}°C")
            print(f"   Max Temp: {sys_perf['temperature']['max_celsius']:.1f}°C")

        print(f"\n📄 Detailed log saved: {log_file}")


def main():
    """Main function with command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Raspberry Pi System Monitor for EdgeFlow"
    )
    parser.add_argument(
        "--htop", action="store_true", help="Run htop-like real-time monitor"
    )
    parser.add_argument("--model", help="Monitor specific model inference")
    parser.add_argument(
        "--inferences", type=int, default=100, help="Number of inferences to run"
    )

    args = parser.parse_args()

    if args.htop:
        run_htop_like_monitor()
    elif args.model:
        if not Path(args.model).exists():
            print(f"❌ Model not found: {args.model}")
            sys.exit(1)
        monitor_inference_with_model(args.model, args.inferences)
    else:
        print("🔍 Raspberry Pi System Monitor for EdgeFlow")
        print("=" * 45)
        print("Usage options:")
        print("  --htop                 : Run real-time system monitor")
        print("  --model <path>         : Monitor model inference performance")
        print("  --inferences <num>     : Number of inferences to run (default: 100)")
        print("\nExamples:")
        print("  python3 pi_system_monitor.py --htop")
        print("  python3 pi_system_monitor.py --model /home/pi/models/model.tflite")


if __name__ == "__main__":
    main()
