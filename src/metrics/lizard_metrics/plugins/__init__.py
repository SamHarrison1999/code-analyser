"""
Expose standard Lizard metric interface and plugin aggregation.
Provides a consistent structure for accessing Lizard metric plugins.
"""

from metrics.lizard_metrics.extractor import get_lizard_extractor

# ✅ Static list of metric names for supervised learning and summary purposes
METRIC_NAME_LIST = [
    "Average function complexity",
    "Average Token Count",
    "Average Parameter Count",
    "Average Function Length",
    "Number of Functions",
    "Maintainability Index"
]

def get_lizard_metric_names() -> list[str]:
    """
    Returns:
        list[str]: Ordered list of Lizard metric names.
    """
    return METRIC_NAME_LIST

def lizard_metric_plugin(file_path: str) -> list[float]:
    """
    Delegates metric extraction to the Lizard plugin-compatible extractor.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        list[float]: Extracted metric values in order of METRIC_NAME_LIST.
    """
    extractor = get_lizard_extractor()
    raw_metrics = extractor(file_path)
    return [entry["value"] for entry in raw_metrics if entry.get("name") in METRIC_NAME_LIST]
