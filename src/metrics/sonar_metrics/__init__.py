# File: code_analyser/src/metrics/sonar_metrics/__init__.py

"""
metrics.sonar_metrics

This subpackage provides SonarQube-based static code metrics
via a plugin-driven architecture. Each plugin extracts a specific metric
from scanner and API data returned by SonarQube.

Features:
- Modular metric extraction using SonarMetricPlugin subclasses
- Dynamic discovery of plugins from the plugins directory
- Ordered, reproducible output for ML pipelines and dashboards
- Confidence and severity metadata for each metric (optional)

Exposes:
- SonarMetricExtractor: Core runner that applies all registered plugins
- gather_sonar_metrics: Returns raw values in plugin order
- gather_sonar_metrics_bundle: Returns metrics with value, confidence, severity
- get_sonar_metric_names: Returns names in extraction order
- load_plugins: Discovers all SonarMetricPlugin subclasses
"""

# ✅ Best Practice: Centralised metric interface for plugin-based SonarQube extraction
# ⚠️ SAST Risk: Improper plugin discovery or missing API tokens may break analysis
# 🧠 ML Signal: This module defines feature vector boundaries for Sonar-based model inputs

from .extractor import SonarMetricExtractor
from .gather import (
    gather_sonar_metrics,
    gather_sonar_metrics_bundle,
    get_sonar_metric_names,
)
from .plugins import load_plugins

# ✅ Best Practice: Define public API clearly for cleaner external access
__all__ = [
    "SonarMetricExtractor",
    "gather_sonar_metrics",
    "gather_sonar_metrics_bundle",
    "get_sonar_metric_names",
    "load_plugins",
]
