"""
metrics.cloc_metrics

This subpackage provides line-based code metrics using the `cloc` tool.

It includes:
- A pluggable plugin architecture for computing metrics from cloc JSON output
- A ClocExtractor class that runs cloc and applies plugins
- A gather_cloc_metrics() function that returns an ordered list of metric values

These metrics are useful for:
- Static analysis
- Training machine learning models
- Generating CSV-compatible summaries

Plugin ordering:
The following metrics are extracted in this fixed order to ensure
consistent CSV column alignment and ML feature stability:

    1. number_of_comments
    2. number_of_lines
    3. number_of_source_lines_of_code
    4. comment_density

Maintaining this order is critical for reproducible ML training.
"""

from .extractor import ClocExtractor, gather_cloc_metrics

__all__ = ["ClocExtractor", "gather_cloc_metrics"]
