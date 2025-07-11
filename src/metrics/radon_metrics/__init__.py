"""
Radon Metrics Subpackage

Provides extraction of code complexity and Halstead metrics using Radon.
Useful for code complexity analysis, maintainability assessment,
and ML feature extraction.

Exposes:
- run_radon: Core function to run Radon and parse metrics.
- gather_radon_metrics: Helper function to gather metrics as an ordered list.
"""

from metrics.radon_metrics.extractor import run_radon
from metrics.radon_metrics.gather import gather_radon_metrics

__all__ = [
    "run_radon",
    "gather_radon_metrics",
]
