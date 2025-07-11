# metrics/pylint_metrics/plugins/__init__.py

# Import the plugin base class and loader for pylint metric plugins
from .base import PylintMetricPlugin
from .default_plugins import DEFAULT_PLUGINS, load_plugins

# Exported names for wildcard import and plugin discovery
__all__ = [
    "PylintMetricPlugin",
    "DEFAULT_PLUGINS",
    "load_plugins",
]
