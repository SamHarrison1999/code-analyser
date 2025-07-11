"""
metrics.cloc_metrics.gather

Provides a plugin-compatible interface to extract CLOC-based metrics
as an ordered list of values for CSV or machine learning export.
"""

from metrics.cloc_metrics.extractor import ClocExtractor
from metrics.cloc_metrics.plugins import load_plugins


def gather_cloc_metrics(file_path: str) -> list:
    """
    Runs the ClocExtractor and returns metrics in a consistent order.

    Args:
        file_path (str): Path to the Python file to analyse.

    Returns:
        list: Ordered metric values.
    """
    # 🧠 ML Signal: Aggregated, ordered metric values support vectorised training data
    try:
        extractor = ClocExtractor(file_path)
        metrics = extractor.extract()
    except Exception:
        # ⚠️ SAST Risk: Do not let CLOC failures crash GUI/CLI analysis
        metrics = {}

    # Maintain exact plugin registration order
    plugin_order = [plugin.name() for plugin in load_plugins()]
    return [metrics.get(name, 0.0 if "density" in name else 0) for name in plugin_order]


def get_cloc_metric_names() -> list:
    """
    Returns the names of the CLOC metrics in the same order as gather_cloc_metrics.

    Returns:
        list: Ordered metric names.
    """
    # ✅ Best Practice: Keep ordering stable and documented
    return [plugin.name() for plugin in load_plugins()]
