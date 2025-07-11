"""
Pylint Metrics Subpackage

Provides extraction of static code metrics using Pylint analysis.
Useful for code quality assessment, ML feature extraction, and auditing.

Exposes:
- PylintMetricExtractor: Class to run Pylint and parse metrics.
- gather_pylint_metrics: Helper function to gather metrics as dict.
"""

from metrics.pylint_metrics.extractor import PylintMetricExtractor
from metrics.pylint_metrics.gather import gather_pylint_metrics

__all__ = [
    "PylintMetricExtractor",
    "gather_pylint_metrics",
]
