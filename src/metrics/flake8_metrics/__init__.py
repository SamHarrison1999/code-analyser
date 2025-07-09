"""
Flake8-based static style and formatting metrics extractor.
"""

from .extractor import Flake8Extractor, gather_flake8_metrics

__all__ = ["Flake8Extractor", "gather_flake8_metrics"]
