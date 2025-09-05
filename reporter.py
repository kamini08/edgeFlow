"""EdgeFlow reporter placeholder.

Generates reports from optimization and benchmarking results.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)


def generate_report(results: Dict[str, Any], out_path: str) -> None:
    """Generate a simple JSON report (placeholder).

    Args:
        results: Aggregated results dictionary.
        out_path: Output file path for the report.
    """

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("[reporter] Wrote placeholder report to %s", out_path)

