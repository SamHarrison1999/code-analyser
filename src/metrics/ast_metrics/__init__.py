# File: src/metrics/ast_metrics/__init__.py

"""
ast_metrics subpackage

This subpackage contains core logic for extracting AST-based metrics
using a plugin-driven architecture. These metrics are suitable for
machine learning pipelines, code quality evaluation, and static analysis.
"""

from .extractor import ASTMetricExtractor
from .gather import gather_ast_metrics

__all__ = [
    "ASTMetricExtractor",
    "gather_ast_metrics",
]
