# File: code_analyser/src/metrics/pyflakes_metrics/gather.py

"""
Gathers Pyflakes metrics using plugin-based extraction.

Used in ML pipelines and CSV/tabular reports.
"""

import logging
from typing import List, Union, Dict
from metrics.pyflakes_metrics.extractor import PyflakesExtractor
from metrics.pyflakes_metrics.plugins import load_plugins

logger = logging.getLogger(__name__)


def gather_pyflakes_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Run Pyflakes on a file and return extracted scalar metrics.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        List[Union[int, float]]: Ordered metric values from plugins.
    """
    try:
        extractor = PyflakesExtractor(file_path)
        result = extractor.extract()
        plugin_order = [plugin.name() for plugin in extractor.plugins]
        return [result.get(name, 0) for name in plugin_order]
    except Exception as e:
        logger.warning(
            f"[gather_pyflakes_metrics] Extraction failed for {file_path}: {type(e).__name__}: {e}"
        )
        return [0 for _ in load_plugins()]


def gather_pyflakes_metrics_bundle(
    file_path: str,
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Returns a list of Pyflakes plugin outputs including:
    metric name, value, confidence, and severity.

    Returns:
        List[Dict]: Structured output for advanced dashboards or ML input.
    """
    try:
        extractor = PyflakesExtractor(file_path)
        results = extractor.extract()

        return [
            {
                "metric": plugin.name(),
                "value": results.get(plugin.name(), 0),
                "confidence": round(plugin.confidence_score(extractor.data), 2),
                "severity": plugin.severity_level(extractor.data),
            }
            for plugin in extractor.plugins
        ]
    except Exception as e:
        logger.warning(
            f"[gather_pyflakes_metrics_bundle] Bundle extraction failed for {file_path}: {type(e).__name__}: {e}"
        )
        return [
            {"metric": plugin.name(), "value": 0, "confidence": 0.0, "severity": "low"}
            for plugin in load_plugins()
        ]


def get_pyflakes_metric_names() -> List[str]:
    """
    Returns the list of Pyflakes plugin metric names in defined order.

    Returns:
        List[str]: Ordered plugin metric names.
    """
    return [plugin.name() for plugin in load_plugins()]
