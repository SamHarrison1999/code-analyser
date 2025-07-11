"""
Plugin system for Pylint metric extraction.

This module defines a placeholder for future plugin extensions
that could enhance or filter Pylint output, transform messages,
or derive secondary metrics.

Provides:
- A central location to register plugin classes
- A loader function to instantiate all registered plugins
"""

# Placeholder base class for future Pylint plugins
class PylintMetricPlugin:
    """
    Base class for Pylint metric plugins.
    Each plugin can inspect or transform Pylint messages.
    """

    def name(self) -> str:
        raise NotImplementedError

    def process(self, messages: list[dict]) -> dict:
        raise NotImplementedError


# No plugins currently implemented
DEFAULT_PLUGINS = []


def load_plugins() -> list[PylintMetricPlugin]:
    """
    Instantiate and return all registered Pylint plugin objects.

    Returns:
        list[PylintMetricPlugin]: A list of plugin instances.
    """
    return [plugin() for plugin in DEFAULT_PLUGINS]


__all__ = [
    "PylintMetricPlugin",
    "DEFAULT_PLUGINS",
    "load_plugins",
]
