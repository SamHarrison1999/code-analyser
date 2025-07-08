# File: src/metrics/__init__.py

"""
metrics package

This package provides static analysis tools for Python code, including
plugin-based AST metric extraction suitable for machine learning workflows.
"""

from .ast_metrics.extractor import ASTMetricExtractor
from .ast_metrics.gather import gather_ast_metrics

# Version metadata for the package
__version__ = "0.1.0"

# Public API
__all__ = [
    "ASTMetricExtractor",
    "gather_ast_metrics",
]
