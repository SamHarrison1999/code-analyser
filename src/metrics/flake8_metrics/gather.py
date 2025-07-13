"""
Flake8 metric gatherer for integration with CLI or CSV pipelines.

Wraps the Flake8Extractor to produce an ordered list of metrics suitable
for structured analysis, ML models, or tabular export.
"""

import logging
from typing import List, Union
from metrics.flake8_metrics.extractor import Flake8Extractor
from metrics.flake8_metrics.plugins import load_plugins

logger = logging.getLogger(__name__)


def gather_flake8_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Extracts Flake8 metrics in a defined order for CSV or ML model usage.

    Args:
        file_path (str): Path to the Python file being analysed.

    Returns:
        List[Union[int, float]]: Ordered list of extracted Flake8 metrics.
    """
    try:
        extractor = Flake8Extractor(file_path)
        metrics = extractor.extract()
        plugin_order = [plugin.name() for plugin in extractor.plugins]
        return [metrics.get(name, 0) for name in plugin_order]

    except Exception as e:
        logger.warning(f"[gather_flake8_metrics] Flake8 extraction failed for {file_path}: {type(e).__name__}: {e}")
        return [0 for _ in get_flake8_metric_names()]  # Ensure consistent shape


def get_flake8_metric_names() -> List[str]:
    """
    Returns the ordered list of metric names matching gather_flake8_metrics output.

    Returns:
        List[str]: Metric names corresponding to Flake8 metrics.
    """
    return [plugin.name() for plugin in load_plugins()]
