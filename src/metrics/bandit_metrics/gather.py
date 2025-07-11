from typing import List
from metrics.bandit_metrics.extractor import BanditExtractor
from metrics.bandit_metrics.plugins import load_plugins


def gather_bandit_metrics(file_path: str) -> List[int]:
    """
    Extracts Bandit metrics from a Python file and returns them
    as an ordered list of values ready for ML models or CSV export.

    The ordering corresponds to the plugin registration order.

    Args:
        file_path (str): Path to the Python file to scan.

    Returns:
        List[int]: Plugin metrics in registered order.
    """
    extractor = BanditExtractor(file_path)
    metric_dict = extractor.extract()
    plugin_order = [plugin.name() for plugin in extractor.plugins]

    # Return values in plugin registration order, with fallback
    return [metric_dict.get(name, 0) for name in plugin_order]
