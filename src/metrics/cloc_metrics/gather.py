from typing import List
from metrics.cloc_metrics.extractor import ClocExtractor
from metrics.cloc_metrics.plugins import load_plugins

def gather_cloc_metrics(file_path: str) -> List[int]:
    """
    Collects all dynamically discovered CLOC metrics for a given file.

    Returns:
        List[int]: List of metric values in plugin load order.
    """
    extractor = ClocExtractor(file_path)
    metric_dict = extractor.extract()
    plugin_order = [p.name() for p in extractor.plugins]
    return [int(metric_dict.get(name, 0)) for name in plugin_order]

def get_cloc_metric_names() -> List[str]:
    """Returns all CLOC metric names in the same order as `gather_cloc_metrics`."""
    return [p.name() for p in load_plugins()]
