# File: src/metrics/bandit_metrics/__init__.py

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

from .extractor import gather_bandit_metrics

__all__ = ["gather_bandit_metrics"]
