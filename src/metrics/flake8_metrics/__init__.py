"""
metrics.flake8_metrics

This subpackage provides Flake8-based static style and formatting metrics
using a plugin-driven architecture. Each plugin extracts a specific metric
from Flake8 output.

Features:
- Modular metric extraction via Flake8MetricPlugin classes
- Dynamic plugin discovery from the plugins directory
- Ordered, reproducible metric output for ML pipelines or CSV export
- Seamless integration with the broader code analyser system

Exposes:
- Flake8Extractor: core runner that applies all discovered plugins
- gather_flake8_metrics: returns metrics as an ordered list of values
- get_flake8_metric_names: returns metric names in extraction order
- load_plugins: dynamically loads all available Flake8MetricPlugin instances
"""

# ✅ Best Practice: Centralise public API for Flake8 metrics using consistent plugin imports
# ⚠️ SAST Risk: Static references to missing modules (e.g., default_plugins) can break integration pipelines
# 🧠 ML Signal: This file defines feature extraction boundaries for the style metric family

from .extractor import Flake8Extractor
from .gather import gather_flake8_metrics, get_flake8_metric_names
from .plugins import load_plugins

__all__ = [
    "Flake8Extractor",
    "gather_flake8_metrics",
    "get_flake8_metric_names",
    "load_plugins",
]
