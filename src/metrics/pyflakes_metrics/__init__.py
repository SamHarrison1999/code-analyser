# File: metrics/pyflakes_metrics/__init__.py

"""
Initialise the Pyflakes metric extractor module.

Exposes key classes and functions for integration with the main metric runner.
"""

from .extractor import PyflakesExtractor, extract_pyflakes_metrics

__all__ = ["PyflakesExtractor", "extract_pyflakes_metrics"]
