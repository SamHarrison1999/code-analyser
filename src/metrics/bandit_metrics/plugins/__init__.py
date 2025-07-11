"""
Plugin system for Bandit metric extraction.

This module exposes all available Bandit plugin classes and provides:
- A central registry of default plugin classes
- A loader function to instantiate all registered plugins
"""

from .base import BanditMetricPlugin
from .default_plugins import (
    HighSeverityIssues,
    MediumSeverityIssues,
    LowSeverityIssues,
    UndefinedSeverityIssues,
)
from .cwe_plugin import (
    CWEFrequencyPlugin,
    MostFrequentCWEPlugin,
    CWENameFrequencyPlugin,
    MostFrequentCWEWithNamePlugin,
)

# ✅ Ordered list of all default plugin classes
DEFAULT_PLUGINS: list[type[BanditMetricPlugin]] = [
    HighSeverityIssues,
    MediumSeverityIssues,
    LowSeverityIssues,
    UndefinedSeverityIssues,
    CWEFrequencyPlugin,
    MostFrequentCWEPlugin,
    CWENameFrequencyPlugin,
    MostFrequentCWEWithNamePlugin,
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
    "CWENameFrequencyPlugin",
    "MostFrequentCWEWithNamePlugin",
    "DEFAULT_PLUGINS",
    "load_plugins",
]
