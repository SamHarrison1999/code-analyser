"""
Flake8 metric gatherer for integration with CLI or CSV pipelines.

Wraps the Flake8Extractor to produce an ordered list of metrics suitable
for structured analysis, ML models, or tabular export.
"""

from typing import Any
from metrics.flake8_metrics.extractor import Flake8Extractor


def gather_flake8_metrics(file_path: str) -> list[Any]:
    """
    Extracts Flake8 metrics in a defined order for CSV or ML model usage.

    Args:
        file_path (str): Path to the Python file being analysed.

    Returns:
        list[Any]: Ordered list of extracted Flake8 metrics.
    """
    # 🧠 ML Signal: Structured and repeatable output helps maintain stable ML feature vectors
    try:
        extractor = Flake8Extractor(file_path)
        metrics = extractor.extract()
    except Exception as e:
        # ⚠️ SAST Risk: Ensure extractor failure does not break entire analysis flow
        from logging import warning
        warning(f"[gather_flake8_metrics] Flake8 extraction failed for {file_path}: {e}")
        metrics = {}

    return [
        metrics.get("number_of_unused_variables", 0),
        metrics.get("number_of_unused_imports", 0),
        metrics.get("number_of_inconsistent_indentations", 0),
        metrics.get("number_of_trailing_whitespaces", 0),
        metrics.get("number_of_long_lines", 0),
        metrics.get("number_of_doc_string_issues", 0),
        metrics.get("number_of_naming_issues", 0),
        metrics.get("number_of_whitespace_issues", 0),
        metrics.get("average_line_length", 0.0),
        metrics.get("number_of_styling_warnings", 0),
        metrics.get("number_of_styling_errors", 0),
        metrics.get("number_of_styling_issues", 0),
    ]


def get_flake8_metric_names() -> list[str]:
    """
    Returns the ordered list of metric names matching gather_flake8_metrics output.

    Returns:
        list[str]: Metric names corresponding to Flake8 metrics.
    """
    # ✅ Best Practice: Keep names stable and clearly mapped to extractor output
    return [
        "number_of_unused_variables",
        "number_of_unused_imports",
        "number_of_inconsistent_indentations",
        "number_of_trailing_whitespaces",
        "number_of_long_lines",
        "number_of_doc_string_issues",
        "number_of_naming_issues",
        "number_of_whitespace_issues",
        "average_line_length",
        "number_of_styling_warnings",
        "number_of_styling_errors",
        "number_of_styling_issues",
    ]
