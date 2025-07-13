"""
Gathers Radon metrics for integration with the code analyser.

Wraps the RadonExtractor to return structured metrics for ML and CSV export.
"""

import logging
from typing import List, Union
from metrics.radon_metrics.extractor import RadonExtractor
from metrics.radon_metrics.plugins import load_plugins

logger = logging.getLogger(__name__)


def gather_radon_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Extracts Radon metrics using dynamic plugin loaders.

    Args:
        file_path (str): Path to the source file.

    Returns:
        List[Union[int, float]]: Extracted metric values.
    """
    try:
        extractor = RadonExtractor(file_path)
        parsed = extractor.extract()
        results = []
        for plugin in load_plugins():
            try:
                value = plugin.extract(parsed, file_path)
            except TypeError:
                # Backwards-compatible fallback if plugin expects only 1 arg
                value = plugin.extract(parsed)
            results.append(value)
        return results

    except Exception as e:
        logger.warning(f"[gather_radon_metrics] Failed to extract from {file_path}: {type(e).__name__}: {e}")
        return [0 for _ in load_plugins()]


def get_radon_metric_names() -> List[str]:
    """
    Returns the names of Radon metrics in the output order.

    Returns:
        List[str]: Ordered metric names.
    """
    return [plugin.name() for plugin in load_plugins()]
