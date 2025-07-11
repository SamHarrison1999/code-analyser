# File: metrics/pyflakes_metrics/gather.py

from typing import Union

def gather_pyflakes_metrics(file_path: str) -> list[Union[int, float]]:
    from metrics.pyflakes_metrics.extractor import PyflakesExtractor  # local import to avoid circular import

    results = PyflakesExtractor(file_path).extract()
    return [
        results.get("number_of_undefined_names", 0),
        results.get("number_of_syntax_errors", 0)
    ]
