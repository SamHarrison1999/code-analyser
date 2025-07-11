"""
Initialise the Pydocstyle metric extractor module.

This package enables docstring compliance checks using pydocstyle.
It provides a plugin-compatible extractor class for integration
with ML pipelines or static analysis tools.
"""

from .extractor import PydocstyleExtractor

__all__ = ["PydocstyleExtractor"]
