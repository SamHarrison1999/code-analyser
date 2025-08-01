# File: code_analyser/src/metrics/cloc_metrics/gather.py

from typing import List, Dict, Union

# ✅ Best Practice: Use consistent imports from plugin namespace to avoid dependency on static paths
# ⚠️ SAST Risk: Avoid hardcoded plugin references to ensure safe dynamic discovery
# 🧠 ML Signal: Feature vector layout here defines model input for supervised learning

from metrics.cloc_metrics.extractor import ClocExtractor
from metrics.cloc_metrics.plugins import load_plugins


def gather_cloc_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Extracts CLOC metrics from a file or directory and returns them
    as an ordered list of values for model input or export.

    The ordering matches the plugin registration order.

    Args:
        file_path (str): Path to a file or directory to analyse.

    Returns:
        List[int | float]: Metric values in registered plugin order.
    """
    extractor = ClocExtractor(file_path)
    metric_dict = extractor.extract()
    plugin_order = [plugin.name() for plugin in extractor.plugins]

    # 🧠 ML Signal: Fixed ordering ensures consistent feature vectors across datasets
    # ✅ Best Practice: Fallback to 0 to avoid alignment errors
    return [metric_dict.get(name, 0) for name in plugin_order]


def gather_cloc_metrics_bundle(file_path: str) -> List[Dict[str, object]]:
    """
    Returns a bundled list of CLOC plugin outputs including:
    value, confidence, and severity for each metric.

    Each item is a dict with metadata for AI analysis or GUI rendering.

    Example:
        [
            {"metric": "number_of_comments", "value": 14, "confidence": 1.0, "severity": "low"},
            ...
        ]

    Args:
        file_path (str): Path to a file or directory to analyse.

    Returns:
        List[Dict[str, object]]: List of metric bundles with metadata.
    """
    extractor = ClocExtractor(file_path)
    metric_values = extractor.extract()
    plugin_order = [plugin.name() for plugin in extractor.plugins]

    return [
        {
            "metric": plugin.name(),
            "value": metric_values.get(plugin.name(), 0),
            "confidence": round(plugin.confidence_score(extractor.data), 2),
            "severity": plugin.severity_level(extractor.data),
        }
        for plugin in extractor.plugins
    ]


def get_cloc_metric_names() -> List[str]:
    """
    Returns the list of CLOC metric names in the order returned by gather_cloc_metrics.

    Returns:
        List[str]: Ordered names of CLOC metric plugins.
    """
    return [plugin.name() for plugin in load_plugins()]
