# File: src/metrics/ast_metrics/plugins/__init__.py

"""
Plugin system for AST metric extraction.

This module exposes available plugin classes and provides a single
import point for registering or loading metric plugins.
"""

from .base import ASTMetricPlugin
from .default_plugins import (
    ModuleDocstringPlugin,
    TodoCommentPlugin,
    NestedFunctionPlugin,
)

__all__ = [
    "ASTMetricPlugin",
    "ModuleDocstringPlugin",
    "TodoCommentPlugin",
    "NestedFunctionPlugin",
]
