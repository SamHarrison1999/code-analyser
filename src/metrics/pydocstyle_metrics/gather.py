"""
Gatherer for Pydocstyle metrics using a plugin-driven architecture.
"""

import logging
from typing import List, Union
from metrics.pydocstyle_metrics.extractor import PydocstyleExtractor
from metrics.pydocstyle_metrics.plugins import load_plugins


def gather_pydocstyle_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Run pydocstyle and apply all registered plugins.

    Args:
        file_path (str): Path to file being analysed.

    Returns:
        List[Union[int, float]]: Ordered list of plugin metric values.
    """
    try:
        raw_output = PydocstyleExtractor(file_path).extract()
        return [plugin.extract(raw_output, file_path) for plugin in load_plugins()]
    except Exception as e:
        logging.warning(f"[gather_pydocstyle_metrics] Extraction failed for {file_path}: {type(e).__name__}: {e}")
        return [0.0 for _ in load_plugins()]


def get_pydocstyle_metric_names() -> List[str]:
    """
    Returns:
        List[str]: Names of metrics in plugin order.
    """
    return [plugin.name() for plugin in load_plugins()]
