"""
Lizard metric plugin interface for ML pipelines or CSV export.

Provides:
- lizard_metric_plugin(): returns extracted metrics as ordered list of floats
- get_lizard_metric_names(): returns names in same order
"""

import logging
from typing import List

from metrics.lizard_metrics.plugins import load_plugins
from metrics.lizard_metrics.extractor import get_lizard_extractor


def get_lizard_metric_names() -> List[str]:
    """
    Returns:
        List[str]: Names of all registered Lizard metric plugins in extraction order.
    """
    return [plugin.name() for plugin in load_plugins()]


def lizard_metric_plugin(file_path: str) -> List[float]:
    """
    Computes Lizard metrics for a given file using plugin architecture.

    Args:
        file_path (str): Path to Python file.

    Returns:
        List[float]: Extracted metrics in fixed order.
    """
    try:
        extractor = get_lizard_extractor()
        raw_metrics = extractor(file_path)
        return [float(plugin.extract(raw_metrics, file_path)) for plugin in load_plugins()]
    except Exception as e:
        logging.warning(f"[lizard_metric_plugin] Lizard extraction failed for {file_path}: {e}")
        return [0.0 for _ in load_plugins()]
