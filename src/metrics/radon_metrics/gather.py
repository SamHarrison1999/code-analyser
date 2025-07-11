from typing import Any
from .extractor import run_radon


def gather_radon_metrics(file_path: str) -> list[Any]:
    """
    Gathers Radon metrics from the given file.

    Returns a list of metrics in the following order:
    - number_of_logical_lines
    - number_of_blank_lines
    - number_of_doc_strings
    - average_halstead_volume
    - average_halstead_difficulty
    - average_halstead_effort
    """
    radon_results = run_radon(file_path)
    return [
        radon_results.get("number_of_logical_lines", 0),
        radon_results.get("number_of_blank_lines", 0),
        radon_results.get("number_of_doc_strings", 0),
        radon_results.get("average_halstead_volume", 0.0),
        radon_results.get("average_halstead_difficulty", 0.0),
        radon_results.get("average_halstead_effort", 0.0),
    ]

def get_radon_metric_names() -> list[str]:
    """
    Returns the list of Radon metric names in the order returned by gather_radon_metrics.
    """
    return [
        "number_of_logical_lines",
        "number_of_blank_lines",
        "number_of_doc_strings",
        "average_halstead_volume",
        "average_halstead_difficulty",
        "average_halstead_effort",
    ]
