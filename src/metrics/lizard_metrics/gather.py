# File: metrics/lizard_metrics/gather.py

"""
Gathers Lizard metrics for unified interface and plugin integration.
"""

from typing import Union
from . import LizardExtractor


def gather_lizard_metrics(file_path: str) -> list[Union[int, float]]:
    """
    Gathers Lizard metrics from the given file.

    Returns:
        list[Union[int, float]]: Ordered metrics:
            - average_cyclomatic_complexity
            - average_token_count
            - total_function_count
            - max_cyclomatic_complexity
            - average_parameters
    """
    results = LizardExtractor(file_path).extract()

    return [
        results.get("average_cyclomatic_complexity", 0.0),
        results.get("average_token_count", 0.0),
        results.get("total_function_count", 0),
        results.get("max_cyclomatic_complexity", 0),
        results.get("average_parameters", 0.0),
    ]
