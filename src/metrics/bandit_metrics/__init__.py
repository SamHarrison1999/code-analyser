"""
Bandit Metrics Subpackage

This subpackage provides static security metrics using Bandit,
a source code analysis tool designed to detect common Python
security issues.

Metrics are extracted through:
- Subprocess execution of the Bandit CLI
- Pluggable plugin system (BanditMetricPlugin)
- gather_bandit_metrics() for ML/CSV-ready output
"""

# ✅ Best Practice: Centralise access to all submodule exports and avoid referencing deleted modules
# ⚠️ SAST Risk: Broken imports can halt static analysis in production CI/CD systems
# 🧠 ML Signal: Tracks which metric systems expose plugin loaders for dynamic analysis pipelines

from .extractor import BanditExtractor
from .gather import gather_bandit_metrics, get_bandit_metric_names
from .plugins import BanditMetricPlugin, load_plugins

__all__ = [
    "BanditExtractor",
    "gather_bandit_metrics",
    "get_bandit_metric_names",
    "BanditMetricPlugin",
    "load_plugins",
]
