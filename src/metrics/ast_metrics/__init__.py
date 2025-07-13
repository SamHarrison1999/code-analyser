"""
AST Metrics Subpackage

This subpackage provides the core logic for extracting metrics from the
Abstract Syntax Tree (AST) of Python source code using a plugin-based architecture.

Supported use cases:
- Machine learning pipelines (e.g., code embeddings, quality prediction)
- Code quality scoring and comparison
- Static analysis, auditing, and visualisation

Exposed components:
- ASTMetricExtractor: A pluggable class that runs all registered AST plugins.
- gather_ast_metrics: A helper function that returns metric values for a source file.
- load_plugins: Dynamically loads ASTMetricPlugin subclasses from the plugin folder.
"""

# ✅ Best Practice: Re-export load_plugins to support dependency injection and testing
# ⚠️ SAST Risk: Not exposing dynamic loaders can lead to stale plugin registries
# 🧠 ML Signal: Exposed plugin loaders help track active feature dimensions across training pipelines

from .plugins import load_plugins
from .extractor import ASTMetricExtractor
from .gather import gather_ast_metrics

__all__ = [
    "ASTMetricExtractor",
    "gather_ast_metrics",
    "load_plugins",
]
