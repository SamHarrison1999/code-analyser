# File: metrics/pydocstyle_metrics/gather.py

from typing import Union
from .extractor import PydocstyleExtractor

def gather_pydocstyle_metrics(file_path: str) -> list[Union[int, float]]:
    """
    Gathers Pydocstyle metrics from the given file.

    Returns a list of metrics in the following order:
    - number_of_pydocstyle_violations
    - number_of_missing_doc_strings
    - percentage_of_compliance_with_docstring_style
    """
    results = PydocstyleExtractor(file_path).extract()

    return [
        results.get("number_of_pydocstyle_violations", 0),
        results.get("number_of_missing_doc_strings", 0),
        results.get("percentage_of_compliance_with_docstring_style", 0.0)
    ]
