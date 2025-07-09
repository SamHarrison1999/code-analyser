"""
metrics.cloc_metrics.plugins

Defines plugin-style metric extractors for CLOC metrics. Each plugin
should expose a `.name()` and `.extract(cloc_data)` method for consistency
with Bandit-style plugin execution.
"""

from .comment_count import CommentCountPlugin
from .total_lines import TotalLinesPlugin
from .source_lines import SourceLinesPlugin
from .comment_density import CommentDensityPlugin

def load_plugins():
    """
    Loads all available CLOC metric plugins.

    Returns:
        list: List of plugin instances.
    """
    return [
        CommentCountPlugin(),
        TotalLinesPlugin(),
        SourceLinesPlugin(),
        CommentDensityPlugin()
    ]
