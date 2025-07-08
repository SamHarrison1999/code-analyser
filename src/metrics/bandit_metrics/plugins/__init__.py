# File: src/metrics/bandit_metrics/plugins/__init__.py

"""
Plugin system for Bandit metric extraction.

This module exposes all available Bandit plugin classes and provides:
- A central location to register default plugin classes
- A loader function to instantiate all registered plugins
"""

from .base import BanditMetricPlugin
from .default_plugins import (
    HighSeverityIssues,
    MediumSeverityIssues,
    LowSeverityIssues,
    UndefinedSeverityIssues,
)
from .cwe_plugin import CWEFrequencyPlugin, MostFrequentCWEPlugin

# Ordered list of all core severity plugins
DEFAULT_PLUGINS: list[type[BanditMetricPlugin]] = [
    HighSeverityIssues,
    MediumSeverityIssues,
    LowSeverityIssues,
    UndefinedSeverityIssues,
    CWEFrequencyPlugin,
    MostFrequentCWEPlugin,
]


def load_plugins() -> list[BanditMetricPlugin]:
    """
    Instantiate and return all registered Bandit plugin objects.

    Returns:
        list[BanditMetricPlugin]: A list of active plugin instances.
    """
    return [plugin() for plugin in DEFAULT_PLUGINS]


__all__ = [
    "BanditMetricPlugin",
    "HighSeverityIssues",
    "MediumSeverityIssues",
    "LowSeverityIssues",
    "UndefinedSeverityIssues",
    "CWEFrequencyPlugin",
    "MostFrequentCWEPlugin",
    "DEFAULT_PLUGINS",
    "load_plugins",
]
