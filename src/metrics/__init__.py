# File: metrics/__init__.py

"""
Top-level package initializer for the `metrics` module.

Exposes core extractors and gatherers for:
- AST metrics
- Bandit (security) metrics
- Cloc (lines/comments) metrics
- Flake8 (style/lint) metrics
- Lizard (complexity/maintainability) metrics
- Pydocstyle (docstring compliance) metrics
- Pyflakes (undefined names/syntax errors) metrics
- Unified metric aggregation
"""

from metrics.ast_metrics.extractor import ASTMetricExtractor
from metrics.bandit_metrics.extractor import BanditExtractor
from metrics.cloc_metrics.extractor import ClocExtractor
from metrics.flake8_metrics.extractor import Flake8Extractor
from metrics.lizard_metrics.extractor import LizardExtractor, extract_lizard_metrics
from metrics.pydocstyle_metrics.extractor import PydocstyleExtractor
from metrics.pyflakes_metrics.extractor import PyflakesExtractor, extract_pyflakes_metrics

from metrics.ast_metrics.gather import gather_ast_metrics
from metrics.bandit_metrics.gather import gather_bandit_metrics
from metrics.cloc_metrics.gather import gather_cloc_metrics
from metrics.flake8_metrics.gather import gather_flake8_metrics
from metrics.lizard_metrics.gather import gather_lizard_metrics
from metrics.pydocstyle_metrics.gather import gather_pydocstyle_metrics
from metrics.pyflakes_metrics.gather import gather_pyflakes_metrics

from metrics.gather import gather_all_metrics, get_all_metric_names

__all__ = [
    "ASTMetricExtractor",
    "BanditExtractor",
    "ClocExtractor",
    "Flake8Extractor",
    "LizardExtractor",
    "PydocstyleExtractor",
    "PyflakesExtractor",
    "extract_lizard_metrics",
    "extract_pyflakes_metrics",
    "gather_ast_metrics",
    "gather_bandit_metrics",
    "gather_cloc_metrics",
    "gather_flake8_metrics",
    "gather_lizard_metrics",
    "gather_pydocstyle_metrics",
    "gather_pyflakes_metrics",
    "gather_all_metrics",
    "get_all_metric_names",
]
