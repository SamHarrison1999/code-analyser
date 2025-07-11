"""
Plugin system for Radon metric extraction.

This module exposes all available Radon plugin classes and provides:
- A central location to register default plugin classes
- A loader function to instantiate all registered plugins
"""

from .base import RadonMetricPlugin
from .default_plugins import (
    LogicalLinesPlugin,
    BlankLinesPlugin,
    DocstringLinesPlugin,
    AverageHalsteadVolumePlugin,
    AverageHalsteadDifficultyPlugin,
    AverageHalsteadEffortPlugin,
)

DEFAULT_PLUGINS = [
    LogicalLinesPlugin,
    BlankLinesPlugin,
    DocstringLinesPlugin,
    AverageHalsteadVolumePlugin,
    AverageHalsteadDifficultyPlugin,
    AverageHalsteadEffortPlugin,
]


def load_plugins() -> list[RadonMetricPlugin]:
    """
    Instantiate and return all registered Radon plugin objects.

    Returns:
        list[RadonMetricPlugin]: A list of active plugin instances.
    """
    return [plugin() for plugin in DEFAULT_PLUGINS]


__all__ = [
    "RadonMetricPlugin",
    "LogicalLinesPlugin",
    "BlankLinesPlugin",
    "DocstringLinesPlugin",
    "AverageHalsteadVolumePlugin",
    "AverageHalsteadDifficultyPlugin",
    "AverageHalsteadEffortPlugin",
    "DEFAULT_PLUGINS",
    "load_plugins",
]
