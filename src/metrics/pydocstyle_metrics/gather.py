"""
Pydocstyle Metric Gatherer

Provides a plugin-compatible interface to extract Pydocstyle-based
documentation metrics in a consistent order for use in ML pipelines or CSV output.
"""

from typing import Union, List
from .extractor import PydocstyleExtractor
import logging


def gather_pydocstyle_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Gathers Pydocstyle metrics from the given file and returns them
    in a fixed order for consistency across data processing pipelines.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[Union[int, float]]: Ordered metrics:
            - number_of_pydocstyle_violations (int)
            - number_of_missing_doc_strings (int)
            - percentage_of_compliance_with_docstring_style (float)
    """
    try:
        results = PydocstyleExtractor(file_path).extract()
    except Exception as e:
        # ⚠️ SAST Risk: Don't allow metric gathering to break pipeline
        logging.warning(f"[gather_pydocstyle_metrics] Extraction failed for {file_path}: {e}")
        results = {}

    return [
        results.get("number_of_pydocstyle_violations", 0),
        results.get("number_of_missing_doc_strings", 0),
        results.get("percentage_of_compliance_with_docstring_style", 0.0),
    ]


def get_pydocstyle_metric_names() -> list[str]:
    """
    Returns the ordered list of Pydocstyle metric names corresponding
    to the output of gather_pydocstyle_metrics.

    Returns:
        list[str]: Metric names.
    """
    return [
        "number_of_pydocstyle_violations",
        "number_of_missing_doc_strings",
        "percentage_of_compliance_with_docstring_style",
    ]
