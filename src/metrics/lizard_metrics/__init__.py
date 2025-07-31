# File: code_analyser/src/metrics/lizard_metrics/__init__.py

"""
metrics.lizard_metrics

This subpackage provides Lizard-based complexity and structural metrics
via a plugin-driven architecture. Each plugin extracts a specific metric
from function-level or file-level Lizard output.

Features:
- Modular metric extraction via LizardMetricPlugin classes
- Dynamic plugin discovery from the plugins directory
- Ordered, reproducible metric output for ML pipelines or CSV export
- Seamless integration with the broader code analyser system

Exposes:
- LizardExtractor: core runner that applies all discovered plugins
- gather_lizard_metrics: returns metrics as an ordered list of values
- gather_lizard_metrics_bundle: returns metrics with value, confidence, severity
- get_lizard_metric_names: returns metric names in extraction order
- load_plugins: dynamically loads all available LizardMetricPlugin instances
"""

# ✅ Best Practice: Centralise public API for Lizard metrics using consistent plugin imports
# ⚠️ SAST Risk: Static references or non-defensive fallbacks may cause pipeline breakage
# 🧠 ML Signal: This module defines structural and complexity-based feature inputs

from .extractor import LizardExtractor
from .gather import (
    gather_lizard_metrics,
    gather_lizard_metrics_bundle,
    get_lizard_metric_names,
)
from .plugins import load_plugins

__all__ = [
    "LizardExtractor",
    "gather_lizard_metrics",
    "gather_lizard_metrics_bundle",
    "get_lizard_metric_names",
    "load_plugins",
]
