"""
Flake8 plugin interface. Exposes the extractor and gatherer
for consistency with plugin-based metric loaders.
"""

from metrics.flake8_metrics.extractor import Flake8Extractor
from metrics.flake8_metrics.gather import gather_flake8_metrics

__all__ = ["Flake8Extractor", "gather_flake8_metrics"]
