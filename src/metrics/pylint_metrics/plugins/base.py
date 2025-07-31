# File: code_analyser/src/metrics/pylint_metrics/plugins/base.py

"""
Base class for Pylint metric plugins.

Each plugin inspects the parsed Pylint output and extracts one metric.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict


class PylintMetricPlugin(ABC):
    """
    Abstract base class for Pylint metric plugins.

    Plugins operate on Pylint diagnostic output and compute a single metric,
    such as a count, score, or category-specific value.
    """

    # ✅ Best Practice: Allow plugins to declare their name and tags explicitly
    plugin_name: str = ""
    plugin_tags: List[str] = []

    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: Unique name of the metric produced by this plugin.
        """
        raise NotImplementedError("Plugin must implement name()")

    @abstractmethod
    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> Any:
        """
        Computes the plugin-specific metric from parsed Pylint results.

        Args:
            pylint_output (List[Dict[str, Any]]): Parsed JSON-like Pylint diagnostics.
            file_path (str): Path to the file being analysed.

        Returns:
            Any: Metric value (typically int or float).
        """
        raise NotImplementedError("Plugin must implement extract()")

    def confidence_score(self, pylint_output: List[Dict[str, Any]]) -> float:
        """
        Optionally provide a confidence score (range 0.0–1.0).

        Returns:
            float: Confidence in the extracted metric.
        """
        return 1.0

    def severity_level(self, pylint_output: List[Dict[str, Any]]) -> str:
        """
        Optionally categorise the severity of the metric result.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        return "low"
