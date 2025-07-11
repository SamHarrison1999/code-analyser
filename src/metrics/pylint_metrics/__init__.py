"""
Pylint Metrics Subpackage

This subpackage provides the core logic for extracting code quality metrics
using Pylint, enabling structured analysis of Python source files.

These metrics are useful for:
- Machine learning pipelines
- Code quality scoring
- Static analysis and audit tools

Exposes:
- gather_pylint_metrics: returns list of metric values for training/inference
"""

from .extractor import gather_pylint_metrics

__all__ = [
    "gather_pylint_metrics",
]
