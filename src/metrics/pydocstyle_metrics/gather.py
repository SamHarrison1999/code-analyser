"""
Pydocstyle Metric Gatherer

Provides a plugin-compatible interface to extract Pydocstyle-based
documentation metrics in a consistent order for use in ML pipelines or CSV output.
"""

from typing import Union
from .extractor import PydocstyleExtractor


def gather_pydocstyle_metrics(file_path: str) -> list[Union[int, float]]:
    """
    Gathers Pydocstyle metrics from the given file and returns them
    in a fixed order for consistency across data processing pipelines.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        list[Union[int, float]]: Ordered metrics:
            - number_of_pydocstyle_violations (int)
            - number_of_missing_doc_strings (int)
            - percentage_of_compliance_with_docstring_style (float)
    """
    try:
        results = PydocstyleExtractor(file_path).extract()
    except Exception:
        # ⚠️ SAST Risk: Don't allow metric gathering to break pipeline
        results = {}

    return [
        results.get("number_of_pydocstyle_violations", 0),
        results.get("number_of_missing_doc_strings", 0),
        results.get("percentage_of_compliance_with_docstring_style", 0.0)
    ]
