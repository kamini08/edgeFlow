#!/usr/bin/env python
"""Test script for CLI visual improvements."""

import time
from cli_formatter import (
    CLIFormatter,
    ProgressBar,
    Spinner,
    Color,
    create_summary_box,
    get_edgeflow_ascii_art,
)


def test_ascii_art():
    """Test EdgeFlow ASCII art display."""
    print("\n=== Testing ASCII Art ===\n")

    formatter = CLIFormatter()
    ascii_art = get_edgeflow_ascii_art().format(version="0.1.0")

    # Display in cyan
    print(formatter.colorize(ascii_art, Color.BRIGHT_CYAN))


def test_headers():
    """Test header formatting."""
    print("\n=== Testing Headers ===\n")

    formatter = CLIFormatter()

    print(formatter.header("Level 1 Header", level=1))
    print("\nSome content under level 1")

    print(formatter.header("Level 2 Header", level=2))
    print("\nSome content under level 2")

    print(formatter.header("Level 3 Header", level=3))
    print("\nSome content under level 3")


def test_messages():
    """Test message formatting."""
    print("\n=== Testing Message Types ===\n")

    formatter = CLIFormatter()

    print(formatter.success("Operation completed successfully"))
    print(formatter.error("An error occurred during processing"))
    print(formatter.warning("This is a warning message"))
    print(formatter.info("This is an informational message"))

    print("\nWith bold text:")
    print(formatter.bold("This text is bold"))


def test_progress_bar():
    """Test progress bar."""
    print("\n=== Testing Progress Bar ===\n")

    formatter = CLIFormatter()
    progress = ProgressBar(100, "Processing files", formatter=formatter)

    for i in range(100):
        progress.update(1, f"File {i+1}/100")
        time.sleep(0.02)  # Simulate work

    progress.finish("All files processed")
    print()


def test_spinner():
    """Test spinner animation."""
    print("\n=== Testing Spinner ===\n")

    formatter = CLIFormatter()

    # Test successful operation
    spinner = Spinner("Loading configuration", formatter)
    spinner.start()

    for i in range(20):
        spinner.update(f"Step {i+1}/20")
        time.sleep(0.1)

    spinner.stop(True, "Configuration loaded")

    # Test failed operation
    spinner = Spinner("Connecting to server", formatter)
    spinner.start()

    for i in range(15):
        spinner.update("Attempting connection...")
        time.sleep(0.1)

    spinner.stop(False, "Connection failed")


def test_stats_formatting():
    """Test statistics formatting."""
    print("\n=== Testing Stats Formatting ===\n")

    formatter = CLIFormatter()

    stats = {
        "model_size_mb": 25.3,
        "latency_ms": 12.5,
        "accuracy_percent": 94.2,
        "optimization_enabled": True,
        "device": "Raspberry Pi 4",
        "quantization": "int8",
    }

    print(formatter.format_stats(stats, "Model Statistics"))


def test_comparison():
    """Test before/after comparison."""
    print("\n=== Testing Comparison Display ===\n")

    formatter = CLIFormatter()

    before = {
        "size_mb": 100.0,
        "latency_ms": 50.0,
        "memory_mb": 200.0,
        "throughput_fps": 10.0
    }

    after = {
        "size_mb": 25.0,
        "latency_ms": 12.0,
        "memory_mb": 50.0,
        "throughput_fps": 40.0
    }

    print(formatter.format_comparison(before, after))


def test_summary_box():
    """Test summary box creation."""
    print("\n=== Testing Summary Box ===\n")

    formatter = CLIFormatter()

    content = [
        "Model size reduction: 75.0%",
        "Latency improvement: 60.0%",
        "Memory reduction: 50.0%",
        "Throughput improvement: 4.0x",
        "",
        "Optimization: SUCCESSFUL",
        "Output: model_optimized.tflite"
    ]

    print(create_summary_box("EdgeFlow Optimization Summary", content, formatter))


def test_color_disabled():
    """Test with colors disabled."""
    print("\n=== Testing Without Colors ===\n")

    formatter = CLIFormatter(use_colors=False)

    print(formatter.header("Header without colors", level=1))
    print(formatter.success("Success without colors"))
    print(formatter.error("Error without colors"))
    print(formatter.warning("Warning without colors"))
    print(formatter.info("Info without colors"))


if __name__ == "__main__":
    print("EdgeFlow CLI Visual Improvements Test Suite")
    print("=" * 50)

    test_ascii_art()
    test_headers()
    test_messages()
    test_spinner()
    test_progress_bar()
    test_stats_formatting()
    test_comparison()
    test_summary_box()
    test_color_disabled()

    print("\n" + "=" * 50)
    print("All tests completed!")