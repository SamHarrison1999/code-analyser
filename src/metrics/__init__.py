# File: src/metrics/__init__.py

"""
metrics package

Provides pluggable metric extractors for static code analysis, including:
- AST-based metrics
- Bandit security metrics
- CLOC line-based metrics

This package is designed to support CLI tools, GUIs, and ML pipelines.
"""

from metrics.ast_metrics.extractor import ASTMetricExtractor
from metrics.bandit_metrics.extractor import BanditExtractor
from metrics.cloc_metrics.extractor import ClocExtractor

from metrics.gather import gather_all_metrics, get_all_metric_names

__all__ = [
    "ASTMetricExtractor",
    "BanditExtractor",
    "ClocExtractor",
    "gather_all_metrics",
    "get_all_metric_names"
]
