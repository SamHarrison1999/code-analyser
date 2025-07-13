"""
Gathers Lizard metrics for unified interface and plugin integration.

Provides a wrapper around the LizardExtractor to return metrics in
a fixed, documented order suitable for ML pipelines or CSV export.
"""

import logging
from typing import Union, List
from metrics.lizard_metrics.extractor import LizardExtractor


def gather_lizard_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Extracts Lizard metrics in a defined, stable order.

    Args:
        file_path (str): Path to the Python file to analyse.

    Returns:
        List[Union[int, float]]: Ordered metrics:
            - average_cyclomatic_complexity
            - average_token_count
            - total_function_count
            - max_cyclomatic_complexity
            - average_parameters
    """
    # 🧠 ML Signal: Stable vector order supports reproducible ML training
    try:
        extractor = LizardExtractor(file_path)
        results = extractor.extract()
    except Exception as e:
        # ⚠️ SAST Risk: Extraction errors must not crash pipeline
        logging.warning(f"[gather_lizard_metrics] Extraction failed for {file_path}: {type(e).__name__}: {e}")
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
    Returns the names of the Lizard metrics in the expected export order.

    Returns:
        List[str]: Ordered list of metric names.
    """
    # ✅ Best Practice: Keep name ordering consistent for ML/CSV compatibility
    return [
        "average_cyclomatic_complexity",
        "average_token_count",
        "total_function_count",
        "max_cyclomatic_complexity",
        "average_parameters",
    ]
