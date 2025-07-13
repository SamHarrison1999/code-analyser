from typing import List, Union

# ✅ Best Practice: Use consistent imports from plugin namespace to avoid dependency on deleted static files
# ⚠️ SAST Risk: Hardcoded imports from missing modules like `default_plugins` cause runtime crashes
# 🧠 ML Signal: This function’s output defines the feature vector layout for supervised models

from metrics.bandit_metrics.extractor import BanditExtractor
from metrics.bandit_metrics.plugins import load_plugins


def gather_bandit_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Extracts Bandit metrics from a Python file and returns them
    as an ordered list of values ready for ML models or CSV export.

    The ordering corresponds to the plugin registration order.

    Args:
        file_path (str): Path to the Python file to scan.

    Returns:
        List[int | float]: Plugin metrics in registered order.
    """
    extractor = BanditExtractor(file_path)
    metric_dict = extractor.extract()
    plugin_order = [plugin.name() for plugin in extractor.plugins]

    # 🧠 ML Signal: Consistent feature ordering across files is critical for training models
    # ✅ Best Practice: Fallback to 0 for missing metrics to preserve order
    return [metric_dict.get(name, 0) for name in plugin_order]


def get_bandit_metric_names() -> List[str]:
    """
    Returns the list of Bandit metric names in the order returned by gather_bandit_metrics.

    Returns:
        List[str]: Ordered names of metrics extracted by Bandit plugins.
    """
    plugins = load_plugins()
    return [plugin.name() for plugin in plugins]
