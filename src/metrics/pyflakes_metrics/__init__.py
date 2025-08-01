# File: code_analyser/src/metrics/pyflakes_metrics/__init__.py

"""
metrics.pyflakes_metrics

This subpackage provides Pyflakes-based static error metrics using a plugin-driven architecture.

Each plugin extracts a specific error-related metric from Pyflakes diagnostic output.

Features:
- Modular metric extraction via PyflakesMetricPlugin classes
- Dynamic plugin discovery from the plugins directory
- Structured and reproducible output for ML or CSV integration
- Compatible with broader Code Analyser architecture

Exposes:
- PyflakesExtractor: core runner that applies all discovered plugins
- gather_pyflakes_metrics: returns metrics as an ordered list of values
- gather_pyflakes_metrics_bundle: returns metrics with value, confidence, severity
- get_pyflakes_metric_names: returns metric names in extraction order
- load_plugins: dynamically loads all available PyflakesMetricPlugin instances
"""

# ✅ Best Practice: Centralise plugin exposure for Pyflakes metrics
# ⚠️ SAST Risk: Avoid static imports from plugin files to prevent brittle dependencies
# 🧠 ML Signal: Defines plugin feature boundaries for error signal learning

from .extractor import PyflakesExtractor
from .gather import (
    gather_pyflakes_metrics,
    gather_pyflakes_metrics_bundle,
    get_pyflakes_metric_names,
)
from .plugins import load_plugins

__all__ = [
    "PyflakesExtractor",
    "gather_pyflakes_metrics",
    "gather_pyflakes_metrics_bundle",
    "get_pyflakes_metric_names",
    "load_plugins",
]
