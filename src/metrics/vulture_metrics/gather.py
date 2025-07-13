"""
Vulture metric gatherer.

Collects metrics from Vulture's unused code detection using
a plugin-based interface to support extensibility.
"""

import logging
from metrics.vulture_metrics.extractor import VultureExtractor
from metrics.vulture_metrics.plugins import load_plugins
from typing import List


def gather_vulture_metrics(file_path: str) -> List[int]:
    """
    Gather Vulture metrics by applying all active plugins.

    Args:
        file_path (str): Path to the source file.

    Returns:
        List[int]: Ordered list of metric values.
    """
    try:
        extractor = VultureExtractor(file_path)
        unused_items = extractor.extract_items()

        values = []
        for plugin in load_plugins():
            try:
                value = plugin.extract(unused_items)
                logging.debug(f"[Vulture Plugin] {plugin.name()} = {value}")
                values.append(value)
            except Exception as plugin_error:
                logging.warning(f"[Vulture Plugin] Failed to extract {plugin.name()}: {plugin_error}")
                values.append(0)

        return values

    except Exception as e:
        logging.error(f"[Vulture] Failed to gather metrics from {file_path}: {type(e).__name__}: {e}")
        return [0] * len(load_plugins())


def get_vulture_metric_names() -> List[str]:
    """
    Get the ordered list of Vulture metric names.

    Returns:
        List[str]: List of metric names.
    """
    return [plugin.name() for plugin in load_plugins()]
