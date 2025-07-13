"""
Pylint Metrics Subpackage

Provides tools for extracting static code metrics from Python source files
using Pylint analysis and a plugin-driven architecture.

This subpackage is designed for:
- Static code quality auditing
- Supervised machine learning feature extraction
- Maintainability and duplication detection
- CSV export and dashboard integration

Exposes:
- PylintMetricExtractor: Class for running Pylint on a file and extracting metrics.
- gather_pylint_metrics: Function that returns ordered metrics for CSV/ML pipelines.
- load_plugins: Dynamically loads plugin metric extractors.
"""

from metrics.pylint_metrics.extractor import PylintMetricExtractor
from metrics.pylint_metrics.gather import gather_pylint_metrics
from metrics.pylint_metrics.plugins import load_plugins

__all__ = [
    "PylintMetricExtractor",
    "gather_pylint_metrics",
    "load_plugins",
]
