# File: code_analyser/src/metrics/radon_metrics/gather.py

"""
Radon plugin-based metric gatherer for ML pipelines and structured exports.

This module wraps RadonExtractor to return either raw metric values
or structured bundles with confidence/severity annotations.

Features:
- Plugin-dispatched metric extraction
- Confidence and severity tagging for supervised ML
- Stable ordering for tabular export
"""

import logging
from typing import List, Union, Dict, Any
from metrics.radon_metrics.plugins import load_plugins

logger = logging.getLogger(__name__)

# ✅ Best Practice: Provide both raw values and detailed bundles for ML + GUI use
# ⚠️ SAST Risk: Extraction errors must fallback safely to avoid downstream crashes
# 🧠 ML Signal: Bundle structure supports richer annotation and supervised learning


def gather_radon_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Extracts Radon plugin-based metrics as a stable list for CSV/ML.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[Union[int, float]]: List of extracted metric values in plugin order.
    """
    try:
        extractor = RadonExtractor(file_path)
        parsed = extractor.extract()
    except Exception as e:
        logger.warning(
            f"[gather_radon_metrics] Failed to extract for {file_path}: {type(e).__name__}: {e}"
        )
        parsed = {}

    results = []
    for plugin in load_plugins():
        try:
            value = plugin.extract(parsed, file_path)
        except TypeError:
            value = plugin.extract(parsed)
        except Exception as e:
            logger.warning(
                f"[gather_radon_metrics] Plugin '{plugin.name()}' error: {type(e).__name__}: {e}"
            )
            value = 0
        results.append(value)

    return results


def gather_radon_metrics_bundle(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts full metric bundles for each Radon plugin including confidence and severity.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[Dict[str, Any]]: List of structured metric dictionaries.
    """
    try:
        extractor = RadonExtractor(file_path)
        parsed = extractor.extract()
    except Exception as e:
        logger.warning(
            f"[gather_radon_metrics_bundle] Fallback for {file_path}: {type(e).__name__}: {e}"
        )
        parsed = {}

    bundle = []
    for plugin in load_plugins():
        try:
            value = plugin.extract(parsed, file_path)
        except TypeError:
            value = plugin.extract(parsed)
        except Exception as e:
            logger.warning(
                f"[gather_radon_metrics_bundle] Plugin '{plugin.name()}' extract error: {type(e).__name__}: {e}"
            )
            value = 0

        try:
            confidence = round(plugin.confidence_score(parsed), 2)
        except Exception:
            confidence = 1.0  # ✅ Default fallback confidence

        try:
            severity = plugin.severity_level(parsed)
        except Exception:
            severity = "low"  # ✅ Default fallback severity

        bundle.append(
            {
                "metric": plugin.name(),
                "value": value,
                "confidence": confidence,
                "severity": severity,
            }
        )

    return bundle


def get_radon_metric_names() -> List[str]:
    """
    Returns the names of all registered Radon plugin metrics in extraction order.

    Returns:
        List[str]: Ordered list of metric names.
    """
    return [plugin.name() for plugin in load_plugins()]
