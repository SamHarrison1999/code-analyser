from typing import Union
from metrics.pyflakes_metrics.extractor import extract_pyflakes_metrics  # Use extraction function instead of class

def gather_pyflakes_metrics(file_path: str) -> list[Union[int, float]]:
    """
    Gathers Pyflakes metrics from the given file.

    Returns:
        list[Union[int, float]]: Ordered metrics:
            - number_of_undefined_names
            - number_of_syntax_errors
    """
    results = extract_pyflakes_metrics(file_path)

    return [
        results.get("number_of_undefined_names", 0),
        results.get("number_of_syntax_errors", 0)
    ]
