# File: src/metrics/__init__.py

"""
metrics package

This package provides static analysis tooling for Python source code,
including plugin-based extraction of:

- AST structural metrics (functions, classes, docstrings, etc.)
- Bandit security metrics (severity levels, CWE grouping)

These metrics support:
- Machine learning pipelines
- Code quality evaluation
- Static code analysis and audit reporting
"""

from .ast_metrics.extractor import ASTMetricExtractor
from .ast_metrics.gather import gather_ast_metrics
from .bandit_metrics.extractor import gather_bandit_metrics
from .gather import gather_all_metrics, get_all_metric_names

# Package version
__version__ = "0.1.0"

# Public API
__all__ = [
    "ASTMetricExtractor",
    "gather_ast_metrics",
    "gather_bandit_metrics",
    "gather_all_metrics",
    "get_all_metric_names",
]
