"""
Gathers Lizard metrics for unified interface and plugin integration.
"""

from typing import Union, List
from . import LizardExtractor
import logging


def gather_lizard_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Gathers Lizard metrics from the given file.

    Args:
        file_path (str): Path to the source Python file.

    Returns:
        List[Union[int, float]]: Ordered metrics:
            - average_cyclomatic_complexity
            - average_token_count
            - total_function_count
            - max_cyclomatic_complexity
            - average_parameters
    """
    # 🧠 ML Signal: Vector consistency supports stable model inputs
    try:
        results = LizardExtractor(file_path).extract()
    except Exception as e:
        # ⚠️ SAST Risk: Ensure metrics gatherer fails safely without crashing
        logging.warning(f"[gather_lizard_metrics] Extraction failed for {file_path}: {e}")
        results = {}

    return [
        results.get("average_cyclomatic_complexity", 0.0),
        results.get("average_token_count", 0.0),
        results.get("total_function_count", 0),
        results.get("max_cyclomatic_complexity", 0),
        results.get("average_parameters", 0.0),
    ]


def get_lizard_metric_names() -> List[str]:
    """
    Returns the metric names in the same order as gather_lizard_metrics.

    Returns:
        List[str]: Ordered metric names.
    """
    # ✅ Best Practice: Stable naming ensures downstream CSV/ML compatibility
    return [
        "average_cyclomatic_complexity",
        "average_token_count",
        "total_function_count",
        "max_cyclomatic_complexity",
        "average_parameters",
    ]
