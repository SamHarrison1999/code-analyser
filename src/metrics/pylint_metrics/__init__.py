"""
Pylint Metrics Subpackage

Provides tools for extracting static code metrics from Python source files
using Pylint analysis.

This is useful for:
- Static code quality auditing
- Supervised machine learning feature extraction
- Software maintainability evaluations

Exposes:
- PylintMetricExtractor: A class to run Pylint on a file and extract structured metrics.
- gather_pylint_metrics: A helper function that runs extraction and aggregates results.
"""

from metrics.pylint_metrics.extractor import PylintMetricExtractor
from metrics.pylint_metrics.gather import gather_pylint_metrics

__all__ = [
    "PylintMetricExtractor",
    "gather_pylint_metrics",
]
