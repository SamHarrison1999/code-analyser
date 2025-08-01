"""
metrics.pylint_metrics

This subpackage provides Pylint-based static code metrics
via a plugin-driven architecture. Each plugin extracts a specific metric
from parsed Pylint JSON output.

Features:
- Modular metric extraction using PylintMetricPlugin subclasses
- Dynamic discovery of plugins from the plugins directory
- Ordered, reproducible output for ML pipelines and dashboards
- Confidence and severity metadata for each metric (optional)

Exposes:
- PylintMetricExtractor: Core runner that applies all registered plugins
- gather_pylint_metrics: Returns raw values in plugin order
- gather_pylint_metrics_bundle: Returns metrics with value, confidence, severity
- get_pylint_metric_names: Returns names in extraction order
- load_plugins: Discovers all PylintMetricPlugin subclasses
"""

# ✅ Best Practice: Centralised metric interface for plugin-based Pylint extraction
# ⚠️ SAST Risk: Improper plugin discovery can cause pipeline crashes if unguarded
# 🧠 ML Signal: This module defines feature vector boundaries for linter-based model inputs

from .extractor import PylintMetricExtractor
from .gather import (
    gather_pylint_metrics,
    gather_pylint_metrics_bundle,
    get_pylint_metric_names,
)
from .plugins import load_plugins

__all__ = [
    "PylintMetricExtractor",
    "gather_pylint_metrics",
    "gather_pylint_metrics_bundle",
    "get_pylint_metric_names",
    "load_plugins",
]
