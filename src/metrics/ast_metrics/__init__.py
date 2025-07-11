"""
AST Metrics Subpackage

This subpackage provides the core logic for extracting metrics from the
abstract syntax tree (AST) of Python source code using a plugin-based architecture.

These metrics support:
- Machine learning pipelines (e.g., code embeddings, quality prediction)
- Code quality scoring and comparisons
- Static analysis, audits, and visualisations

Exposes:
- ASTMetricExtractor: A pluggable class that runs all registered AST plugins.
- gather_ast_metrics: A helper function that returns metric values for a file.
"""

from .extractor import ASTMetricExtractor
from .gather import gather_ast_metrics

__all__ = [
    "ASTMetricExtractor",
    "gather_ast_metrics",
]
