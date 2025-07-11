"""
Plugin system for AST metric extraction.

This module exposes all available AST plugin classes and provides:
- A central registry of default plugin classes
- A loader function to instantiate all registered plugins
"""

from .base import ASTMetricPlugin
from .default_plugins import (
    ModuleDocstringPlugin,
    TodoCommentPlugin,
    NestedFunctionPlugin,
    LambdaFunctionPlugin,
    MagicMethodPlugin,
    AssertStatementPlugin,
    ClassDocstringPlugin,
)

# List of default plugin classes to register
DEFAULT_PLUGINS = [
    ModuleDocstringPlugin,
    TodoCommentPlugin,
    NestedFunctionPlugin,
    LambdaFunctionPlugin,
    MagicMethodPlugin,
    AssertStatementPlugin,
    ClassDocstringPlugin,
]

def load_plugins() -> list[ASTMetricPlugin]:
    """
    Instantiate and return all registered AST plugin objects.

    Returns:
        list[ASTMetricPlugin]: A list of active plugin instances.
    """
    return [plugin() for plugin in DEFAULT_PLUGINS]


__all__ = [
    "ASTMetricPlugin",
    "ModuleDocstringPlugin",
    "TodoCommentPlugin",
    "NestedFunctionPlugin",
    "LambdaFunctionPlugin",
    "MagicMethodPlugin",
    "AssertStatementPlugin",
    "ClassDocstringPlugin",
    "DEFAULT_PLUGINS",
    "load_plugins",
]
