# File: metrics/pyflakes_metrics/gather.py

from typing import Union
from .extractor import PyflakesExtractor


def gather_pyflakes_metrics(file_path: str) -> list[Union[int, float]]:
    """
    Gathers Pyflakes metrics from the given file.

    Returns:
        list[Union[int, float]]: Ordered metrics:
            - number_of_undefined_names
            - number_of_syntax_errors
    """
    results = PyflakesExtractor(file_path).extract()

    return [
        results.get("number_of_undefined_names", 0),
        results.get("number_of_syntax_errors", 0)
    ]
