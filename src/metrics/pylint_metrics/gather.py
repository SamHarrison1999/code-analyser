from typing import List
from metrics.pylint_metrics.extractor import PylintMetricExtractor

# Fixed order of Pylint message types for consistent output
_METRIC_ORDER = [
    "convention",
    "refactor",
    "warning",
    "error",
    "fatal",
]


def gather_pylint_metrics(file_path: str) -> List[int]:
    """
    Extracts Pylint metrics from a Python file and returns them
    as a list in a fixed order, suitable for ML input or tabular export.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[int]: Ordered list of Pylint message counts by type.
    """
    messages = PylintMetricExtractor(file_path).extract()
    counts = {key: 0 for key in _METRIC_ORDER}

    for msg in messages:
        msg_type = msg.get("type")
        if msg_type in counts:
            counts[msg_type] += 1

    return [counts[key] for key in _METRIC_ORDER]
