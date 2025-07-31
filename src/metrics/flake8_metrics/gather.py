# File: code_analyser/src/metrics/flake8_metrics/gather.py

"""
Flake8 metric gatherer for integration with CLI or CSV pipelines.

Wraps the Flake8Extractor to produce an ordered list of metrics suitable
for structured analysis, ML models, or tabular export.
"""

import logging
from typing import List, Union, Dict
from metrics.flake8_metrics.extractor import Flake8Extractor
from metrics.flake8_metrics.plugins import load_plugins

logger = logging.getLogger(__name__)


# ✅ Best Practice: Standardised extraction pattern for tabular metrics
# ⚠️ SAST Risk: Ensure metrics fallback on plugin failure
# 🧠 ML Signal: Feature vectors from Flake8 aid in predicting stylistic adherence
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
        logger.warning(
            f"[gather_flake8_metrics] Flake8 extraction failed for {file_path}: {type(e).__name__}: {e}"
        )
        return [0 for _ in get_flake8_metric_names()]


def gather_flake8_metrics_bundle(file_path: str) -> List[Dict[str, object]]:
    """
    Returns a structured bundle of metric data with value, confidence, and severity.

    Returns:
        List[Dict[str, object]]: List of metric dictionaries with metadata.
    """
    try:
        extractor = Flake8Extractor(file_path)
        metrics = extractor.extract()

        return [
            {
                "metric": plugin.name(),
                "value": metrics.get(plugin.name(), 0),
                "confidence": round(plugin.confidence_score(extractor.data), 2),
                "severity": plugin.severity_level(extractor.data),
            }
            for plugin in extractor.plugins
        ]
    except Exception as e:
        logger.warning(
            f"[gather_flake8_metrics_bundle] Failed to bundle metrics for {file_path}: {type(e).__name__}: {e}"
        )
        return [
            {"metric": plugin.name(), "value": 0, "confidence": 0.0, "severity": "low"}
            for plugin in load_plugins()
        ]


def get_flake8_metric_names() -> List[str]:
    """
    Returns the ordered list of metric names matching gather_flake8_metrics output.

    Returns:
        List[str]: Metric names corresponding to Flake8 metrics.
    """
    return [plugin.name() for plugin in load_plugins()]
