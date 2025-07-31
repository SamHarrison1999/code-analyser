"""
AST Metrics Subpackage

This subpackage provides the core logic for extracting metrics from the
Abstract Syntax Tree (AST) of Python source code using a plugin-based architecture.

Supported use cases:
- Machine learning pipelines (e.g., code embeddings, quality prediction)
- Code quality scoring and comparison
- Static analysis, auditing, and visualisation
- TensorBoard/GUI overlays with severity + confidence export

Exposed components:
- ASTMetricExtractor: A pluggable class that runs all registered AST plugins.
- gather_ast_metrics: Extracts ordered metric values.
- gather_ast_metrics_bundle: Returns metrics with confidence/severity overlays.
- get_ast_metric_names: Fixed ordering for ML alignment.
- load_plugins: Dynamically loads ASTMetricPlugin subclasses from the plugin folder.
"""

# ✅ Best Practice: Re-export all key components for shared ML/GUI use
# ⚠️ SAST Risk: Not exposing updated bundle/score APIs limits auditability
# 🧠 ML Signal: Bundled metrics w/ severity and confidence form the feature foundation

from .plugins import load_plugins
from .extractor import ASTMetricExtractor
from .gather import (
    gather_ast_metrics,
    gather_ast_metrics_bundle,
    get_ast_metric_names,
    gather_ast_confidence_scores,
    gather_ast_severity_levels,
)

__all__ = [
    "ASTMetricExtractor",
    "gather_ast_metrics",
    "gather_ast_metrics_bundle",
    "get_ast_metric_names",
    "gather_ast_confidence_scores",
    "gather_ast_severity_levels",
    "load_plugins",
]
