from typing import List, Dict
from metrics.pylint_metrics.extractor import PylintMetricExtractor


def gather_pylint_metrics(file_path: str) -> Dict[str, int]:
    """
    Extract Pylint metrics from the given file and return as a dictionary
    of metric name to count. Metrics keys are prefixed with 'pylint_'.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        Dict[str, int]: Dictionary of Pylint metrics counts.
    """
    extractor = PylintMetricExtractor(file_path)
    return extractor.extract()
