import logging
from pathlib import Path
from typing import Any

from . import run_lizard


def gather(file_path: Path) -> list[dict[str, Any]]:
    """
    Gathers code metrics using Lizard for the given file.

    Parameters:
        file_path (Path): The path to the file to analyse.

    Returns:
        list[dict[str, Any]]: A list of metric dictionaries, each containing:
            - name (str): Metric name
            - value (Any): Metric value
            - units (str | None): Unit of measurement (if applicable)
            - success (bool): Whether the metric was successfully computed
            - error (str | None): Error message if any
    """
    try:
        return run_lizard(str(file_path))
    except Exception as e:
        logging.error(f"Failed to gather Lizard metrics for {file_path}: {e}")
        return [{
            "name": "lizard_metrics_failure",
            "value": None,
            "units": None,
            "success": False,
            "error": str(e),
        }]
