"""
Initialise the pydocstyle metric plugin interface.

Provides access to:
- Metric name list for supervised ML and CSV headers
- Plugin-compatible gatherer function
"""

from metrics.pydocstyle_metrics.gather import gather_pydocstyle_metrics

# ✅ Stable list of metric names for downstream compatibility
METRIC_NAME_LIST = [
    "number_of_pydocstyle_violations",
    "number_of_missing_doc_strings",
    "percentage_of_compliance_with_docstring_style"
]


def get_pydocstyle_metric_names() -> list[str]:
    """
    Returns the ordered list of pydocstyle metric names.

    Returns:
        list[str]: Metric names in output order.
    """
    return METRIC_NAME_LIST


def pydocstyle_metric_plugin(file_path: str) -> list[float]:
    """
    Run the pydocstyle metric extractor and return metrics in stable order.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        list[float]: Metric values in the order defined by METRIC_NAME_LIST.
    """
    try:
        return gather_pydocstyle_metrics(file_path)
    except Exception:
        # ⚠️ SAST Risk: Gracefully handle unexpected extractor failures
        return [0.0 for _ in METRIC_NAME_LIST]
