# File: src/metrics/ast_metrics/__init__.py

"""
AST Metrics Subpackage

This subpackage provides the core logic for extracting AST-based metrics
from Python source code using a plugin-driven architecture.

These metrics are useful for:
- Machine learning pipelines
- Code quality scoring
- Static analysis and audit tools

Exposes:
- ASTMetricExtractor: a pluggable metric runner
- gather_ast_metrics: returns list of metric values for training/inference
"""

from .extractor import ASTMetricExtractor
from .gather import gather_ast_metrics

__all__ = [
    "ASTMetricExtractor",
    "gather_ast_metrics",
]
