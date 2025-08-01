# File: code_analyser/src/metrics/cloc_metrics/__init__.py

from .extractor import ClocExtractor
from .gather import (
    gather_cloc_metrics,
    gather_cloc_metrics_bundle,
    get_cloc_metric_names,
)
from .plugins import load_plugins, ClocMetricPlugin

# ✅ Best Practice: Centralised exports simplify dynamic module loading and plugin system reuse
__all__ = [
    "ClocExtractor",
    "gather_cloc_metrics",
    "gather_cloc_metrics_bundle",
    "get_cloc_metric_names",
    "ClocMetricPlugin",
    "load_plugins",
]
