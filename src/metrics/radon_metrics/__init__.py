# File: code_analyser/src/metrics/radon_metrics/__init__.py

"""
metrics.radon_metrics

This subpackage provides Radon-based static code metrics
via a plugin-driven architecture. Each plugin extracts a specific metric
from pre-parsed Radon raw and Halstead data.

Features:
- Modular metric extraction using RadonMetricPlugin subclasses
- Dynamic discovery of plugins from the plugins directory
- Ordered, reproducible output for ML pipelines and dashboards
- Confidence and severity metadata for each metric (optional)

Exposes:
- RadonMetricExtractor: Core runner that applies all registered plugins
- gather_radon_metrics: Returns raw values in plugin order
- gather_radon_metrics_bundle: Returns metrics with value, confidence, severity
- get_radon_metric_names: Returns names in extraction order
- load_plugins: Discovers all RadonMetricPlugin subclasses
"""

# ✅ Best Practice: Centralised metric interface for plugin-based Radon extraction
# ⚠️ SAST Risk: Improper plugin discovery can cause pipeline crashes if unguarded
# 🧠 ML Signal: This module defines feature vector boundaries for Radon-based model inputs

from .extractor import RadonMetricExtractor
from .gather import (
    gather_radon_metrics,
    gather_radon_metrics_bundle,
    get_radon_metric_names,
)
from .plugins import load_plugins

__all__ = [
    "RadonMetricExtractor",
    "gather_radon_metrics",
    "gather_radon_metrics_bundle",
    "get_radon_metric_names",
    "load_plugins",
]
