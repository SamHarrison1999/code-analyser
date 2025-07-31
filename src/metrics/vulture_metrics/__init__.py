# File: code_analyser/src/metrics/vulture_metrics/__init__.py

"""
metrics.vulture_metrics

This subpackage provides Vulture-based static code metrics
via a plugin-driven architecture. Each plugin extracts a specific metric
from raw unused code items discovered by the Vulture tool.

Features:
- Modular metric extraction using VultureMetricPlugin subclasses
- Dynamic discovery of plugins from the plugins directory
- Ordered, reproducible output for ML pipelines and dashboards
- Confidence and severity metadata for each metric (optional)

Exposes:
- VultureMetricExtractor: Core runner that applies all registered plugins
- gather_vulture_metrics: Returns raw values in plugin order
- gather_vulture_metrics_bundle: Returns metrics with value, confidence, severity
- get_vulture_metric_names: Returns names in extraction order
- load_plugins: Discovers all VultureMetricPlugin subclasses
"""

# ✅ Best Practice: Centralised metric interface for plugin-based Vulture extraction
# ⚠️ SAST Risk: Improper plugin discovery can cause pipeline crashes if unguarded
# 🧠 ML Signal: This module defines feature vector boundaries for unused-code models

from .extractor import VultureMetricExtractor
from .gather import (
    gather_vulture_metrics,
    gather_vulture_metrics_bundle,
    get_vulture_metric_names,
)
from .plugins import load_plugins

__all__ = [
    "VultureMetricExtractor",
    "gather_vulture_metrics",
    "gather_vulture_metrics_bundle",
    "get_vulture_metric_names",
    "load_plugins",
]
