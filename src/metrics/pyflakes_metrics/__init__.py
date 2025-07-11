"""
Initialise the Pyflakes metric extractor module.

Exposes key classes and functions for integration with the main metric runner.
"""

from metrics.pyflakes_metrics.extractor import PyflakesExtractor, extract_pyflakes_metrics

__all__ = ["PyflakesExtractor", "extract_pyflakes_metrics"]
