# File: code_analyser/src/metrics/lizard_metrics/gather.py

"""
Gathers Lizard metrics for unified interface and plugin integration.

Provides a wrapper around the LizardExtractor to return metrics in
a fixed, documented order suitable for ML pipelines or CSV export.
"""

import logging
from typing import Union, List, Dict, Any
from metrics.lizard_metrics.extractor import LizardExtractor
from metrics.lizard_metrics.plugins import load_plugins


# ✅ Best Practice: Stable metric order supports feature vector reproducibility
# ⚠️ SAST Risk: Always return consistent structure even if Lizard fails
# 🧠 ML Signal: Bundle provides rich input for supervised learning


def gather_lizard_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Extracts Lizard metrics in a defined, stable order.

    Args:
        file_path (str): Path to the Python file to analyse.

    Returns:
        List[Union[int, float]]: Ordered metric values.
    """
    # 🧠 ML Signal: Stable vector order supports reproducible ML training
    try:
        extractor = LizardExtractor(file_path)
        metric_dict = extractor.extract()
        plugin_order = [plugin.name() for plugin in extractor.plugins]
        return [metric_dict.get(name, 0) for name in plugin_order]
    except Exception as e:
        # ⚠️ SAST Risk: Extraction errors must not crash pipeline
        logging.warning(
            f"[gather_lizard_metrics] Extraction failed for {file_path}: {type(e).__name__}: {e}"
        )
        return [0 for _ in get_lizard_metric_names()]


def gather_lizard_metrics_bundle(file_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of Lizard plugin outputs including value, confidence, and severity.

    Args:
        file_path (str): Path to the Python file to analyse.

    Returns:
        List[Dict[str, object]]: List of dictionaries with metric metadata.
    """
    try:
        extractor = LizardExtractor(file_path)
        metric_values = extractor.extract()
        return [
            {
                "metric": plugin.name(),
                "value": metric_values.get(plugin.name(), 0),
                "confidence": round(plugin.confidence_score(extractor.data), 2),
                "severity": plugin.severity_level(extractor.data),
            }
            for plugin in extractor.plugins
        ]
    except Exception as e:
        logging.warning(
            f"[gather_lizard_metrics_bundle] Failed to bundle metrics for {file_path}: {type(e).__name__}: {e}"
        )
        return [
            {"metric": plugin.name(), "value": 0, "confidence": 0.0, "severity": "low"}
            for plugin in load_plugins()
        ]


def get_lizard_metric_names() -> List[str]:
    """
    Returns the names of the Lizard metrics in the extraction order.

    Returns:
        List[str]: Ordered metric names.
    """
    return [plugin.name() for plugin in load_plugins()]
