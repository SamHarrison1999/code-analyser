# File: metrics/gather.py

"""
Aggregates metrics from various extractors into a single unified list.
Used by CLI, GUI, and model export.

Included extractors:
- AST
- Bandit (security)
- Cloc (lines/comments)
- Flake8 (style/lint)
- Lizard (complexity/maintainability)
- Pydocstyle (docstring compliance)
- Pyflakes (undefined names, syntax errors)
"""

import tempfile
from typing import List, Union

from metrics.ast_metrics.extractor import ASTMetricExtractor
from metrics.bandit_metrics.extractor import BanditExtractor
from metrics.cloc_metrics.extractor import ClocExtractor
from metrics.flake8_metrics.extractor import Flake8Extractor
from metrics.lizard_metrics.extractor import LizardExtractor, extract_lizard_metrics
from metrics.pydocstyle_metrics.extractor import PydocstyleExtractor
from metrics.pyflakes_metrics.extractor import extract_pyflakes_metrics


def gather_all_metrics(file_path: str) -> List[Union[int, float]]:
    """
    Gathers all metric values from AST, Bandit, Cloc, Flake8, Lizard, Pydocstyle, and Pyflakes extractors.

    Args:
        file_path (str): Path to the Python file to analyze.

    Returns:
        list[int | float]: Unified list of all extracted metrics.
    """
    ast_metrics = ASTMetricExtractor(file_path).extract()
    bandit_metrics = BanditExtractor(file_path).extract()
    cloc_metrics = ClocExtractor(file_path).extract()
    flake8_metrics = Flake8Extractor(file_path).extract()
    lizard_metrics = extract_lizard_metrics(file_path)
    pydocstyle_metrics = PydocstyleExtractor(file_path).extract()
    pyflakes_metrics = extract_pyflakes_metrics(file_path)

    return (
        list(ast_metrics.values()) +
        list(bandit_metrics.values()) +
        list(cloc_metrics.values()) +
        list(flake8_metrics.values()) +
        list(lizard_metrics.values()) +
        list(pydocstyle_metrics.values()) +
        list(pyflakes_metrics.values())
    )


def get_all_metric_names() -> List[str]:
    """
    Returns the names of all metrics in the same order as gather_all_metrics.

    Returns:
        list[str]: List of all metric names.
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".py") as f:
        f.write("def foo(): pass")
        f.flush()

        ast_keys = list(ASTMetricExtractor(f.name).extract().keys())
        bandit_keys = list(BanditExtractor(f.name).extract().keys())
        cloc_keys = list(ClocExtractor(f.name).extract().keys())
        flake8_keys = list(Flake8Extractor(f.name).extract().keys())
        lizard_keys = list(extract_lizard_metrics(f.name).keys())
        pydocstyle_keys = list(PydocstyleExtractor(f.name).extract().keys())
        pyflakes_keys = list(extract_pyflakes_metrics(f.name).keys())

    return (
        ast_keys +
        bandit_keys +
        cloc_keys +
        flake8_keys +
        lizard_keys +
        pydocstyle_keys +
        pyflakes_keys
    )
