"""
Bandit Metrics Subpackage

This subpackage provides static security metrics using Bandit,
a source code analysis tool designed to detect common Python
security issues.

Metrics are extracted through:
- subprocess execution of Bandit CLI
- pluggable plugin system (BanditMetricPlugin)
- gather_bandit_metrics() for ML/CSV-ready output
"""

from .extractor import BanditExtractor
from .gather import gather_bandit_metrics  # Correct import from gather.py

__all__ = [
    "BanditExtractor",
    "gather_bandit_metrics",
]
