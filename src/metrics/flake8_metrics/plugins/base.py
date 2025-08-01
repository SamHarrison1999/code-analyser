# File: code_analyser/src/metrics/flake8_metrics/plugins/base.py

"""
Base class for Flake8 metric plugins.

Each plugin must define:
- name(): a unique metric name
- extract(): a method to compute the metric from parsed Flake8 output
"""

from abc import ABC, abstractmethod
from typing import Any, List


class Flake8MetricPlugin(ABC):
    """
    Abstract base class for Flake8 metric plugins.

    All Flake8 plugins must:
    - Provide a unique metric name via `name()`
    - Implement `extract()` to compute a metric from Flake8 diagnostics
    - Optionally provide confidence and severity metadata
    """

    # ✅ Best Practice: Plugin metadata fields for registry introspection
    plugin_name: str = ""
    plugin_tags: list[str] = []

    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: The unique name of the plugin metric (used as a key in output dictionaries).
        """
        raise NotImplementedError("Plugin must implement name()")

    @abstractmethod
    def extract(self, flake8_output: List[str], file_path: str) -> Any:
        """
        Computes a metric from the Flake8 output lines.

        Args:
            flake8_output (List[str]): Parsed Flake8 diagnostic lines.
            file_path (str): Path to the source file being analysed.

        Returns:
            Any: The computed metric value (typically int or float).
        """
        raise NotImplementedError("Plugin must implement extract()")

    def confidence_score(self, flake8_output: List[str]) -> float:
        """
        Optionally return a confidence score for this metric (0.0–1.0).

        Returns:
            float: Confidence score (defaults to 1.0).
        """
        return 1.0

    def severity_level(self, flake8_output: List[str]) -> str:
        """
        Optionally classify this metric’s result severity level.

        Returns:
            str: One of 'low', 'medium', or 'high' (defaults to 'low').
        """
        return "low"
