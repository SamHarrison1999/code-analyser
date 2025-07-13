"""
Gathers Pyflakes metrics using plugin-based extraction.

Used in ML pipelines and CSV/tabular reports.
"""

import logging
from typing import List
from metrics.pyflakes_metrics.extractor import PyflakesExtractor
from metrics.pyflakes_metrics.plugins import load_plugins


def gather_pyflakes_metrics(file_path: str) -> List[int]:
    try:
        extractor = PyflakesExtractor(file_path)
        output = extractor.extract()
        return [plugin.extract(output, file_path) for plugin in load_plugins()]
    except Exception as e:
        logging.warning(f"[gather_pyflakes_metrics] Extraction failed for {file_path}: {e}")
        return [0 for _ in load_plugins()]


def get_pyflakes_metric_names() -> List[str]:
    return [plugin.name() for plugin in load_plugins()]
