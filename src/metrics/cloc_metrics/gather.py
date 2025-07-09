"""
metrics.cloc_metrics.gather

Provides a plugin-compatible interface to extract CLOC-based metrics
as an ordered list of values for CSV or machine learning export.
"""

from metrics.cloc_metrics.extractor import ClocExtractor


def gather_cloc_metrics(file_path: str) -> list:
    """
    Runs the ClocExtractor and returns metrics in a consistent order.

    Args:
        file_path (str): Path to the Python file to analyse.

    Returns:
        list: Ordered metric values.
    """
    # 🧠 ML Signal: Aggregated, ordered metric values support vectorized training data
    try:
        extractor = ClocExtractor(file_path)
        metrics = extractor.extract()
    except Exception:
        # ⚠️ SAST Risk: Do not let CLOC failures crash GUI/CLI analysis
        metrics = {}

    return [
        metrics.get("number_of_comments", 0),
        metrics.get("number_of_lines", 0),
        metrics.get("number_of_source_lines_of_code", 0),
        metrics.get("comment_density", 0.0),
    ]


def get_cloc_metric_names() -> list:
    """
    Returns the names of the CLOC metrics in the same order as gather_cloc_metrics.

    Returns:
        list: Ordered metric names.
    """
    # ✅ Best Practice: Keep ordering stable and documented
    return [
        "number_of_comments",
        "number_of_lines",
        "number_of_source_lines_of_code",
        "comment_density"
    ]
