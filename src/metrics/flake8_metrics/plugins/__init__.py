"""
metrics.flake8_metrics

This subpackage provides Flake8-based static analysis metrics.

It includes:
- A Flake8Extractor class that executes Flake8 on a target file
- A gather_flake8_metrics() function that returns a list of metric values
- Integration with plugin-style metric loaders across the code analyser system

These metrics are useful for:
- Static analysis pipelines
- Machine learning feature extraction
- Visualisation in GUI/CSV reports

Returned metrics include:
- Total number of Flake8 issues
- Issue types grouped by Flake8 category (e.g. E, F, W)

Example usage:
    >>> from metrics.flake8_metrics import gather_flake8_metrics
    >>> gather_flake8_metrics("my_script.py")
    [12, 5, 3, 4]  # e.g. total, E, F, W counts
"""

from metrics.flake8_metrics.extractor import Flake8Extractor
from metrics.flake8_metrics.gather import gather_flake8_metrics

__all__ = ["Flake8Extractor", "gather_flake8_metrics"]
