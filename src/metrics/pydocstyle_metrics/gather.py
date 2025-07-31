# File: code_analyser/src/metrics/pydocstyle_metrics/gather.py

"""
Gatherer for Pydocstyle metrics using a plugin-driven architecture.

Provides:
- gather_pydocstyle_metrics(): raw value-only extraction
- gather_pydocstyle_metrics_bundle(): structured result with metadata
- get_pydocstyle_metric_names(): ordered plugin metric names
"""

import logging
from typing import List, Union, Dict
from metrics.pydocstyle_metrics.extractor import PydocstyleExtractor
from metrics.pydocstyle_metrics.plugins import load_plugins


def gather_pydocstyle_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Run pydocstyle and extract value-only plugin metrics.

    Args:
        file_path (str): File path to analyse.

    Returns:
        List[Union[int, float]]: Ordered list of plugin metric values.
    """
    try:
        raw_output = PydocstyleExtractor(file_path).extract()
        return [plugin.extract(raw_output, file_path) for plugin in load_plugins()]
    except Exception as e:
        logging.warning(
            f"[gather_pydocstyle_metrics] Extraction failed for {file_path}: {type(e).__name__}: {e}"
        )
        return [0.0 for _ in load_plugins()]


def gather_pydocstyle_metrics_bundle(
    file_path: str,
) -> List[Dict[str, Union[str, float, int]]]:
    """
    Run pydocstyle and extract structured plugin metrics including metadata.

    Args:
        file_path (str): File path to analyse.

    Returns:
        List[Dict]: Each entry contains 'metric', 'value', 'confidence', and 'severity'.
    """
    try:
        raw_output = PydocstyleExtractor(file_path).extract()
        return [
            {
                "metric": plugin.name(),
                "value": plugin.extract(raw_output, file_path),
                "confidence": round(plugin.confidence_score(raw_output), 2),
                "severity": plugin.severity_level(raw_output),
            }
            for plugin in load_plugins()
        ]
    except Exception as e:
        logging.warning(
            f"[gather_pydocstyle_metrics_bundle] Bundle extraction failed for {file_path}: {type(e).__name__}: {e}"
        )
        return [
            {"metric": plugin.name(), "value": 0, "confidence": 0.0, "severity": "low"}
            for plugin in load_plugins()
        ]


def get_pydocstyle_metric_names() -> List[str]:
    """
    Return ordered metric names matching plugin output order.

    Returns:
        List[str]: Names of all extracted metrics.
    """
    return [plugin.name() for plugin in load_plugins()]
