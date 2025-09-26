"""CLI Formatter Module for EdgeFlow Compiler.

This module provides pretty printing, colored output, and progress bars
for the EdgeFlow compiler CLI to enhance user experience.
"""

import sys
import time
from typing import Any, Dict, List, Optional
from enum import Enum


class Color(Enum):
    """Terminal color codes for pretty output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'

    # Colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright Colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background Colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


class Icons:
    """Text-based indicators for CLI display."""
    SUCCESS = '[SUCCESS]'
    ERROR = '[ERROR]'
    WARNING = '[WARNING]'
    INFO = '[INFO]'
    CHECK = '[✓]'
    CROSS = '[✗]'
    ARROW = '->'
    BULLET = '*'
    GEAR = '[PROCESSING]'
    ROCKET = '[OPTIMIZING]'
    SEARCH = '[ANALYZING]'
    LIGHTNING = '[FAST]'
    CHART = '[METRICS]'
    BRAIN = '[AI]'
    SPARKLES = '[COMPLETE]'
    FIRE = '[HOT]'
    PACKAGE = '[OUTPUT]'
    TIMER = '[TIME]'
    MEMORY = '[MEMORY]'
    CPU = '[CPU]'


class CLIFormatter:
    """Formatter for pretty CLI output."""

    def __init__(self, use_colors: bool = True, use_icons: bool = False):
        """Initialize CLI formatter.

        Args:
            use_colors: Whether to use colored output
            use_icons: Whether to use text indicators (disabled by default for professional look)
        """
        self.use_colors = use_colors and self._supports_color()
        self.use_icons = use_icons

    @staticmethod
    def _supports_color() -> bool:
        """Check if terminal supports color output."""
        if not hasattr(sys.stdout, 'isatty'):
            return False
        if not sys.stdout.isatty():
            return False
        if sys.platform == 'win32':
            return False  # Simplified; could check for Windows 10 with color support
        return True

    def colorize(self, text: str, color: Color) -> str:
        """Apply color to text if colors are enabled.

        Args:
            text: Text to colorize
            color: Color to apply

        Returns:
            Colored text or original text if colors disabled
        """
        if not self.use_colors:
            return text
        return f"{color.value}{text}{Color.RESET.value}"

    def bold(self, text: str) -> str:
        """Make text bold."""
        return self.colorize(text, Color.BOLD)

    def success(self, text: str, with_icon: bool = False) -> str:
        """Format success message."""
        icon = f"{Icons.SUCCESS} " if self.use_icons and with_icon else ""
        return f"{icon}{self.colorize(text, Color.GREEN)}"

    def error(self, text: str, with_icon: bool = False) -> str:
        """Format error message."""
        icon = f"{Icons.ERROR} " if self.use_icons and with_icon else ""
        return f"{icon}{self.colorize(text, Color.RED)}"

    def warning(self, text: str, with_icon: bool = False) -> str:
        """Format warning message."""
        icon = f"{Icons.WARNING} " if self.use_icons and with_icon else ""
        return f"{icon}{self.colorize(text, Color.YELLOW)}"

    def info(self, text: str, with_icon: bool = False) -> str:
        """Format info message."""
        icon = f"{Icons.INFO} " if self.use_icons and with_icon else ""
        return f"{icon}{self.colorize(text, Color.CYAN)}"

    def header(self, text: str, level: int = 1) -> str:
        """Format section header.

        Args:
            text: Header text
            level: Header level (1-3)

        Returns:
            Formatted header
        """
        if level == 1:
            separator = "=" * 60
            formatted = f"\n{self.colorize(separator, Color.BRIGHT_BLUE)}"
            formatted += f"\n{self.bold(self.colorize(text.upper(), Color.BRIGHT_BLUE))}"
            formatted += f"\n{self.colorize(separator, Color.BRIGHT_BLUE)}"
        elif level == 2:
            separator = "-" * 50
            formatted = f"\n{self.colorize(separator, Color.CYAN)}"
            formatted += f"\n{self.colorize(text, Color.BRIGHT_CYAN)}"
            formatted += f"\n{self.colorize(separator, Color.CYAN)}"
        else:
            formatted = f"\n{self.colorize('▶', Color.MAGENTA)} {self.colorize(text, Color.BRIGHT_MAGENTA)}"

        return formatted

    def format_stats(self, stats: Dict[str, Any], title: Optional[str] = None) -> str:
        """Format statistics in a pretty table-like format.

        Args:
            stats: Dictionary of statistics
            title: Optional title for the stats

        Returns:
            Formatted statistics string
        """
        lines = []

        if title:
            lines.append(self.header(title, level=3))

        max_key_len = max(len(str(k)) for k in stats.keys()) if stats else 0

        for key, value in stats.items():
            formatted_key = str(key).replace('_', ' ').title()
            formatted_key = formatted_key.ljust(max_key_len + 2)

            # Format value based on type
            if isinstance(value, float):
                if 'percent' in key.lower() or 'reduction' in key.lower():
                    formatted_value = f"{value:.1f}%"
                    color = Color.GREEN if value > 0 else Color.YELLOW
                elif 'time' in key.lower() or 'latency' in key.lower():
                    formatted_value = f"{value:.2f}ms"
                    color = Color.CYAN
                else:
                    formatted_value = f"{value:.2f}"
                    color = Color.WHITE
            elif isinstance(value, bool):
                formatted_value = "Yes" if value else "No"
                color = Color.GREEN if value else Color.RED
            else:
                formatted_value = str(value)
                color = Color.WHITE

            line = f"  {self.colorize(formatted_key, Color.BRIGHT_WHITE)}: {self.colorize(formatted_value, color)}"
            lines.append(line)

        return '\n'.join(lines)

    def format_comparison(self, before: Dict[str, float], after: Dict[str, float]) -> str:
        """Format before/after comparison.

        Args:
            before: Metrics before optimization
            after: Metrics after optimization

        Returns:
            Formatted comparison string
        """
        lines = []
        lines.append(self.header("Optimization Results", level=2))

        # Calculate improvements
        improvements = []

        for key in before.keys():
            if key in after:
                before_val = before[key]
                after_val = after[key]

                if before_val > 0:
                    if 'size' in key.lower() or 'memory' in key.lower():
                        improvement = ((before_val - after_val) / before_val) * 100
                        if improvement > 0:
                            improvements.append((key, before_val, after_val, improvement))
                    elif 'latency' in key.lower() or 'time' in key.lower():
                        improvement = ((before_val - after_val) / before_val) * 100
                        if improvement > 0:
                            improvements.append((key, before_val, after_val, improvement))
                    elif 'throughput' in key.lower():
                        improvement = ((after_val - before_val) / before_val) * 100
                        if improvement > 0:
                            improvements.append((key, before_val, after_val, improvement))

        # Display improvements
        for metric, before_val, after_val, improvement in improvements:
            formatted_metric = metric.replace('_', ' ').title()

            arrow = self.colorize("→", Color.BRIGHT_YELLOW)
            before_str = self.colorize(f"{before_val:.2f}", Color.DIM)
            after_str = self.colorize(f"{after_val:.2f}", Color.BRIGHT_GREEN)
            improvement_str = self.colorize(f"({improvement:.1f}% improvement)", Color.GREEN)

            line = f"  {formatted_metric}: {before_str} {arrow} {after_str} {improvement_str}"
            lines.append(line)

        return '\n'.join(lines)


class ProgressBar:
    """Simple progress bar for CLI."""

    def __init__(self, total: int, description: str = "", width: int = 50,
                 formatter: Optional[CLIFormatter] = None):
        """Initialize progress bar.

        Args:
            total: Total number of steps
            description: Description of the operation
            width: Width of the progress bar
            formatter: Optional CLI formatter for colored output
        """
        self.total = total
        self.description = description
        self.width = width
        self.current = 0
        self.formatter = formatter or CLIFormatter()
        self.start_time = time.time()

    def update(self, step: int = 1, status: str = "") -> None:
        """Update progress bar.

        Args:
            step: Number of steps to advance
            status: Optional status message
        """
        self.current = min(self.current + step, self.total)
        self._render(status)

    def finish(self, message: str = "Complete") -> None:
        """Finish the progress bar."""
        self.current = self.total
        self._render(message)
        print()  # New line after completion

    def _render(self, status: str = "") -> None:
        """Render the progress bar."""
        if self.total == 0:
            percent = 100
        else:
            percent = (self.current / self.total) * 100

        filled = int(self.width * self.current / max(self.total, 1))
        bar = '█' * filled + '░' * (self.width - filled)

        elapsed = time.time() - self.start_time

        if percent < 100:
            bar_color = Color.YELLOW
            percent_color = Color.BRIGHT_YELLOW
        else:
            bar_color = Color.GREEN
            percent_color = Color.BRIGHT_GREEN

        bar_str = self.formatter.colorize(bar, bar_color)
        percent_str = self.formatter.colorize(f"{percent:3.0f}%", percent_color)
        elapsed_str = self.formatter.colorize(f"[{elapsed:.1f}s]", Color.DIM)

        line = f"\r{self.description}: {bar_str} {percent_str} {elapsed_str}"

        if status:
            line += f" - {self.formatter.colorize(status, Color.CYAN)}"

        sys.stdout.write(line)
        sys.stdout.flush()


class Spinner:
    """Simple spinner for indicating ongoing operations."""

    FRAMES = ['|', '/', '-', '\\']

    def __init__(self, description: str = "Processing",
                 formatter: Optional[CLIFormatter] = None):
        """Initialize spinner.

        Args:
            description: Description of the operation
            formatter: Optional CLI formatter
        """
        self.description = description
        self.formatter = formatter or CLIFormatter()
        self.frame = 0
        self.running = False
        self.start_time = time.time()

    def start(self) -> None:
        """Start the spinner."""
        self.running = True
        self.frame = 0
        self.start_time = time.time()

    def update(self, status: str = "") -> None:
        """Update spinner with optional status."""
        if not self.running:
            return

        frame_char = self.FRAMES[self.frame % len(self.FRAMES)]
        elapsed = time.time() - self.start_time

        line = f"\r{self.formatter.colorize(frame_char, Color.CYAN)} {self.description}"
        line += f" {self.formatter.colorize(f'[{elapsed:.1f}s]', Color.DIM)}"

        if status:
            line += f" - {self.formatter.colorize(status, Color.BRIGHT_CYAN)}"

        sys.stdout.write(line + ' ' * 20)  # Clear remaining characters
        sys.stdout.flush()

        self.frame += 1

    def stop(self, success: bool = True, message: str = "") -> None:
        """Stop the spinner.

        Args:
            success: Whether the operation was successful
            message: Final message to display
        """
        self.running = False
        elapsed = time.time() - self.start_time

        if success:
            icon = '[OK]' if not self.formatter.use_icons else Icons.SUCCESS
            color = Color.GREEN
            default_msg = "Complete"
        else:
            icon = '[FAIL]' if not self.formatter.use_icons else Icons.ERROR
            color = Color.RED
            default_msg = "Failed"

        final_message = message or default_msg

        line = f"\r{self.formatter.colorize(icon, color)} {self.description}"
        line += f" - {self.formatter.colorize(final_message, color)}"
        line += f" {self.formatter.colorize(f'[{elapsed:.1f}s]', Color.DIM)}"

        print(line + ' ' * 20)  # Clear and print final message


def get_edgeflow_ascii_art() -> str:
    """Return EdgeFlow ASCII art logo."""
    return """
    ███████╗██████╗  ██████╗ ███████╗███████╗██╗      ██████╗ ██╗    ██╗
    ██╔════╝██╔══██╗██╔════╝ ██╔════╝██╔════╝██║     ██╔═══██╗██║    ██║
    █████╗  ██║  ██║██║  ███╗█████╗  █████╗  ██║     ██║   ██║██║ █╗ ██║
    ██╔══╝  ██║  ██║██║   ██║██╔══╝  ██╔══╝  ██║     ██║   ██║██║███╗██║
    ███████╗██████╔╝╚██████╔╝███████╗██║     ███████╗╚██████╔╝╚███╔███╔╝
    ╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝

    Edge AI Model Optimization Compiler v{version}
    """


def create_summary_box(title: str, content: List[str],
                       formatter: Optional[CLIFormatter] = None) -> str:
    """Create a formatted summary box.

    Args:
        title: Box title
        content: List of content lines
        formatter: Optional formatter

    Returns:
        Formatted box string
    """
    formatter = formatter or CLIFormatter()

    # Calculate box width
    max_width = max(len(title), max(len(line) for line in content)) + 4
    box_width = min(max_width, 80)  # Cap at 80 chars

    lines = []

    # Top border
    lines.append(formatter.colorize('╔' + '═' * (box_width - 2) + '╗', Color.BRIGHT_BLUE))

    # Title
    title_padding = (box_width - len(title) - 2) // 2
    title_line = '║' + ' ' * title_padding + title + ' ' * (box_width - len(title) - title_padding - 2) + '║'
    lines.append(formatter.colorize(title_line, Color.BRIGHT_BLUE))

    # Separator
    lines.append(formatter.colorize('╠' + '═' * (box_width - 2) + '╣', Color.BRIGHT_BLUE))

    # Content
    for line in content:
        if len(line) > box_width - 4:
            # Wrap long lines
            wrapped = line[:box_width - 7] + '...'
            content_line = f"║ {wrapped} ║"
        else:
            padding = box_width - len(line) - 3
            content_line = f"║ {line}{' ' * padding}║"
        lines.append(formatter.colorize(content_line, Color.WHITE))

    # Bottom border
    lines.append(formatter.colorize('╚' + '═' * (box_width - 2) + '╝', Color.BRIGHT_BLUE))

    return '\n'.join(lines)