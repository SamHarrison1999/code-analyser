# File: metrics/pyflakes_metrics/gather.py

from typing import Union

def gather_pyflakes_metrics(file_path: str) -> list[Union[int, float]]:
    """
    Gathers Pyflakes metrics from the given file.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        list[Union[int, float]]: Ordered metrics:
            - number_of_undefined_names
            - number_of_syntax_errors
    """
    # Local import to avoid circular import issues
    from metrics.pyflakes_metrics.extractor import PyflakesExtractor

    results = PyflakesExtractor(file_path).extract()
    return [
        results.get("number_of_undefined_names", 0),
        results.get("number_of_syntax_errors", 0),
    ]

def get_pyflakes_metric_names() -> list[str]:
    """
    Returns the list of Pyflakes metric names in the same order as gather_pyflakes_metrics.

    Returns:
        list[str]: Ordered metric names.
    """
    return [
        "number_of_undefined_names",
        "number_of_syntax_errors",
    ]
