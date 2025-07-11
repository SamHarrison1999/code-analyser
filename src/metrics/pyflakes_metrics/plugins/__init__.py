# File: metrics/pyflakes_metrics/plugins/__init__.py

"""
Expose standard Pyflakes metric interface and plugin aggregation.
Provides a consistent structure for accessing Pyflakes metric plugins.
"""

from metrics.pyflakes_metrics.gather import gather_pyflakes_metrics

# ✅ Static list of metric names for supervised learning and summary purposes
METRIC_NAME_LIST = [
    "number_of_undefined_names",
    "number_of_syntax_errors",
    "number_of_pyflakes_issues"
]

def get_pyflakes_metric_names() -> list[str]:
    """
    Returns:
        list[str]: Ordered list of Pyflakes metric names.
    """
    return METRIC_NAME_LIST

def pyflakes_metric_plugin(file_path: str) -> list[float]:
    """
    Delegates metric extraction to Pyflakes gatherer.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        list[float]: Extracted metric values in order of METRIC_NAME_LIST.
    """
    return gather_pyflakes_metrics(file_path)
