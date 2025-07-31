# File: code_analyser/src/metrics/pylint_metrics/gather.py

"""
Pylint plugin-based metric gatherer for ML pipelines and structured exports.

This module wraps PylintMetricExtractor to return either raw metric values
or structured bundles with confidence/severity annotations.

Features:
- Plugin-dispatched metric extraction
- Confidence and severity tagging for supervised ML
- Stable ordering for tabular export
"""

import logging
from typing import List, Union, Dict, Any
from metrics.pylint_metrics.extractor import PylintMetricExtractor
from metrics.pylint_metrics.plugins import load_plugins

logger = logging.getLogger(__name__)

# ✅ Best Practice: Provide both raw values and detailed bundles for ML + GUI use
# ⚠️ SAST Risk: Extraction errors must fallback safely to avoid downstream crashes
# 🧠 ML Signal: Bundle structure supports richer annotation and supervised learning


def gather_pylint_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Extracts Pylint plugin-based metrics as a stable list for CSV/ML.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[Union[int, float]]: List of extracted metric values in plugin order.
    """
    try:
        extractor = PylintMetricExtractor(file_path)
        results = extractor.extract()
    except Exception as e:
        logger.warning(
            f"[gather_pylint_metrics] Failed to extract metrics for {file_path}: {type(e).__name__}: {e}"
        )
        results = {}

    return [results.get(plugin.name(), 0) for plugin in load_plugins()]


def gather_pylint_metrics_bundle(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts full metric bundles for each pylint plugin including confidence and severity.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[Dict[str, Any]]: List of structured metric dictionaries.
    """
    try:
        extractor = PylintMetricExtractor(file_path)
        output = extractor.raw_output  # Used for scoring by plugins
        results = extractor.extract()
    except Exception as e:
        logger.warning(
            f"[gather_pylint_metrics_bundle] Fallback for {file_path}: {type(e).__name__}: {e}"
        )
        output = []
        results = {}

    bundle = []
    for plugin in load_plugins():
        try:
            value = results.get(plugin.name(), 0)
            confidence = round(plugin.confidence_score(output), 2)
            severity = plugin.severity_level(output)
        except Exception as e:
            logger.warning(
                f"[gather_pylint_metrics_bundle] Plugin '{plugin.name()}' error: {type(e).__name__}: {e}"
            )
            value, confidence, severity = 0, 0.0, "low"

        bundle.append(
            {
                "metric": plugin.name(),
                "value": value,
                "confidence": confidence,
                "severity": severity,
            }
        )

    return bundle


def get_pylint_metric_names() -> List[str]:
    """
    Returns the names of all registered pylint plugin metrics in extraction order.

    Returns:
        List[str]: Ordered list of metric names.
    """
    return [plugin.name() for plugin in load_plugins()]
