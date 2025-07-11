from typing import Dict, List
from metrics.pylint_metrics.extractor import PylintMetricExtractor


def gather_pylint_metrics(file_path: str) -> Dict[str, int]:
    """
    Extracts Pylint severity metrics from the given file.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        Dict[str, int]: Dictionary containing counts of Pylint severity levels:
                        convention, refactor, warning, error, fatal.
    """
    extractor = PylintMetricExtractor(file_path)
    return extractor.extract()


def get_pylint_metric_names() -> List[str]:
    """
    Returns the list of standard Pylint severity metric keys.

    These correspond to the top-level counts returned by Pylint.

    Returns:
        List[str]: List of severity metric names.
    """
    return [
        "convention",
        "refactor",
        "warning",
        "error",
        "fatal",
    ]
