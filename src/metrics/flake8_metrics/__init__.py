"""
metrics.flake8_metrics

This subpackage provides Flake8-based static style and formatting metrics
using a plugin-driven architecture. Each plugin extracts a specific metric
from Flake8 output.

Exposes:
- Flake8Extractor: core runner that applies all plugins
- gather_flake8_metrics: ordered list of metric values for ML/CSV
"""

from .extractor import Flake8Extractor, gather_flake8_metrics

__all__ = ["Flake8Extractor", "gather_flake8_metrics"]
