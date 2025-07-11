"""
Initialise the Lizard metric extractor module.

Exposes key classes and functions for integration with the main metric runner.
"""

from .extractor import LizardExtractor, extract_lizard_metrics, get_lizard_extractor

__all__ = [
    "LizardExtractor",
    "extract_lizard_metrics",
    "get_lizard_extractor",
]
