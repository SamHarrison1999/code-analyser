"""
Plugin system for CLOC metric extraction.

This module defines all available plugin-style extractors for CLOC output,
used to compute consistent metrics across Python projects.

Each plugin must inherit from `CLOCMetricPlugin` and implement:
- name() -> str — a unique metric name
- extract(cloc_data: dict) -> int | float — computes the value from parsed CLOC JSON

This structure supports:
- Static metric analysis for CSV/ML export
- Extensible plugin discovery and aggregation
"""

from .base import CLOCMetricPlugin
from .comment_count import CommentCountPlugin
from .total_lines import TotalLinesPlugin
from .source_lines import SourceLinesPlugin
from .comment_density import CommentDensityPlugin

# Ordered list of registered plugin classes
DEFAULT_PLUGINS: list[type[CLOCMetricPlugin]] = [
    CommentCountPlugin,
    TotalLinesPlugin,
    SourceLinesPlugin,
    CommentDensityPlugin,
]

def load_plugins() -> list[CLOCMetricPlugin]:
    """
    Instantiate and return all registered CLOC metric plugins.

    Returns:
        list[CLOCMetricPlugin]: List of plugin instances.
    """
    return [plugin() for plugin in DEFAULT_PLUGINS]


__all__ = [
    "CLOCMetricPlugin",
    "CommentCountPlugin",
    "TotalLinesPlugin",
    "SourceLinesPlugin",
    "CommentDensityPlugin",
    "DEFAULT_PLUGINS",
    "load_plugins",
]
