# File: code_analyser/src/metrics/vulture_metrics/gather.py

"""
Vulture plugin-based metric gatherer for ML pipelines and structured exports.

This module wraps VultureExtractor to return either raw metric values
or structured bundles with confidence/severity annotations.

Features:
- Plugin-dispatched metric extraction
- Confidence and severity tagging for supervised ML
- Stable ordering for tabular export
"""

import logging
from typing import List, Union, Dict, Any
from metrics.vulture_metrics.plugins import load_plugins

logger = logging.getLogger(__name__)


# ✅ Best Practice: Provide both raw values and detailed bundles for ML + GUI use
# ⚠️ SAST Risk: Extraction errors must fallback safely to avoid downstream crashes
# 🧠 ML Signal: Bundle structure supports richer annotation and supervised learning


def gather_vulture_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Extracts Vulture plugin-based metrics as a stable list for CSV/ML.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[Union[int, float]]: List of extracted metric values in plugin order.
    """
    try:
        extractor = VultureExtractor(file_path)
        parsed = extractor.extract_items()
    except Exception as e:
        logger.warning(
            f"[gather_vulture_metrics] Failed to extract for {file_path}: {type(e).__name__}: {e}"
        )
        parsed = []

    results = []
    for plugin in load_plugins():
        try:
            value = plugin.extract(parsed)
        except Exception as e:
            logger.warning(
                f"[gather_vulture_metrics] Plugin '{plugin.name()}' error: {type(e).__name__}: {e}"
            )
            value = 0
        results.append(value)

    return results


def gather_vulture_metrics_bundle(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts full metric bundles for each Vulture plugin including confidence and severity.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[Dict[str, Any]]: List of structured metric dictionaries.
    """
    try:
        extractor = VultureExtractor(file_path)
        parsed = extractor.extract_items()
    except Exception as e:
        logger.warning(
            f"[gather_vulture_metrics_bundle] Fallback for {file_path}: {type(e).__name__}: {e}"
        )
        parsed = []

    bundle = []
    for plugin in load_plugins():
        try:
            value = plugin.extract(parsed)
        except Exception as e:
            logger.warning(
                f"[gather_vulture_metrics_bundle] Plugin '{plugin.name()}' extract error: {type(e).__name__}: {e}"
            )
            value = 0

        try:
            confidence = round(plugin.confidence_score(parsed), 2)
        except Exception:
            confidence = 1.0

        try:
            severity = plugin.severity_level(parsed)
        except Exception:
            severity = "low"

        bundle.append(
            {
                "metric": plugin.name(),
                "value": value,
                "confidence": confidence,
                "severity": severity,
            }
        )

    return bundle


def get_vulture_metric_names() -> List[str]:
    """
    Returns the names of all registered Vulture plugin metrics in extraction order.

    Returns:
        List[str]: Ordered list of metric names.
    """
    return [plugin.name() for plugin in load_plugins()]
