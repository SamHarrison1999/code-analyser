import logging
from typing import List, Union
from metrics.pylint_metrics.extractor import PylintMetricExtractor
from metrics.pylint_metrics.plugins import load_plugins


def gather_pylint_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Extracts pylint plugin-based metrics as a stable list for CSV/ML.

    Args:
        file_path (str): Python file path.

    Returns:
        List[Union[int, float]]: List of metric values.
    """
    try:
        extractor = PylintMetricExtractor(file_path)
        results = extractor.extract()
    except Exception as e:
        logging.warning(f"[gather_pylint_metrics] Failed to extract metrics for {file_path}: {e}")
        results = {}

    return [results.get(plugin.name(), 0) for plugin in load_plugins()]


def get_pylint_metric_names() -> List[str]:
    """
    Returns plugin metric names in stable order.

    Returns:
        List[str]: Names of metrics used in gather_pylint_metrics().
    """
    return [plugin.name() for plugin in load_plugins()]
