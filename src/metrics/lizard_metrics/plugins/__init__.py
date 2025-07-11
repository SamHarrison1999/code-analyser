"""
Expose standard Lizard metric interface and plugin aggregation.

Provides a consistent structure for accessing Lizard metric plugins
and exporting metrics in a stable order suitable for ML pipelines
and CSV summarisation.
"""

from typing import List
from metrics.lizard_metrics.extractor import get_lizard_extractor
import logging

# ✅ Static list of plugin-defined metric names for supervised learning
METRIC_NAME_LIST: List[str] = [
    "average_function_complexity",
    "average_token_count",
    "average_parameter_count",
    "average_function_length",
    "number_of_functions",
    "maintainability_index",
]


def get_lizard_metric_names() -> List[str]:
    """
    Returns:
        List[str]: Ordered list of Lizard metric names matching plugin outputs.
    """
    return METRIC_NAME_LIST


def lizard_metric_plugin(file_path: str) -> List[float]:
    """
    Delegates metric extraction to the Lizard plugin-compatible extractor.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        List[float]: Extracted metric values in the order of METRIC_NAME_LIST.
    """
    try:
        extractor = get_lizard_extractor()
        raw_metrics = extractor(file_path)

        # ✅ Best Practice: Map plugin output and preserve order
        metric_map = {entry.get("name"): entry.get("value", 0.0) for entry in raw_metrics}
        return [float(metric_map.get(name, 0.0)) for name in METRIC_NAME_LIST]

    except Exception as e:
        # ⚠️ SAST Risk: Protect pipeline from extractor crashes
        logging.warning(f"[lizard_metric_plugin] Lizard extraction failed for {file_path}: {e}")
        return [0.0 for _ in METRIC_NAME_LIST]
