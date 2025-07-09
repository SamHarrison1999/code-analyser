"""
metrics.cloc_metrics

This package contains logic for extracting line-based code metrics using `cloc`.
It provides a ClocExtractor class and a utility function for retrieving metrics
in list form, consistent with AST and Bandit extractors.
"""

from .extractor import ClocExtractor, gather_cloc_metrics

__all__ = ["ClocExtractor", "gather_cloc_metrics"]
