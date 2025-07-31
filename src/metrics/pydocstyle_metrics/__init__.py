# File: code_analyser/src/metrics/pydocstyle_metrics/__init__.py

"""
metrics.pydocstyle_metrics

This subpackage provides Pydocstyle-based documentation quality metrics
using a dynamic plugin system. Each plugin computes a single docstring-related metric.

Features:
- Modular metric extraction via PydocstyleMetricPlugin classes
- Dynamic plugin discovery from the plugins directory
- Reproducible metric ordering for ML/CSV/export pipelines
- Supports confidence and severity metadata for each metric

Exposes:
- PydocstyleExtractor: core runner that applies all discovered plugins
- extract_pydocstyle_metrics: returns bundled metrics with metadata
- gather_pydocstyle_metrics: returns metrics as an ordered list of values
- get_pydocstyle_metric_names: returns metric names in plugin load order
- load_plugins: dynamically loads all PydocstyleMetricPlugin instances
"""

# ✅ Best Practice: Centralised public API for metrics + plugin registry
# ⚠️ SAST Risk: Static plugin mappings should be avoided — use discovery instead
# 🧠 ML Signal: This module defines the docstring feature vector boundary

from .extractor import (
    PydocstyleExtractor,
    extract_pydocstyle_metrics,
    get_pydocstyle_extractor,
)
from .gather import (
    gather_pydocstyle_metrics,
    gather_pydocstyle_metrics_bundle,
    get_pydocstyle_metric_names,
)
from .plugins import load_plugins

__all__ = [
    "PydocstyleExtractor",
    "extract_pydocstyle_metrics",
    "get_pydocstyle_extractor",
    "gather_pydocstyle_metrics",
    "gather_pydocstyle_metrics_bundle",
    "get_pydocstyle_metric_names",
    "load_plugins",
]
