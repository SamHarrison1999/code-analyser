# File: code_analyser/src/metrics/pyflakes_metrics/plugins/base.py

"""
Base class for Pyflakes metric plugins.

Each plugin inspects parsed Pyflakes output and extracts a specific scalar metric.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class PyflakesMetricPlugin(ABC):
    """
    Abstract base class for Pyflakes metric plugins.

    Plugins must:
    - Define a unique metric name
    - Extract a scalar value (typically int or float) from Pyflakes diagnostics
    - Optionally provide confidence and severity annotations
    """

    # ✅ Best Practice: Metadata for plugin discovery and filtering
    plugin_name: str = ""
    plugin_tags: List[str] = []

    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of the metric provided by this plugin.

        Returns:
            str: Metric name used as a dictionary key.
        """
        raise NotImplementedError("Plugin must implement name()")

    @abstractmethod
    def extract(self, pyflakes_output: List[str], file_path: str) -> Any:
        """
        Compute the metric from the provided Pyflakes diagnostic output.

        Args:
            pyflakes_output (List[str]): Raw Pyflakes output lines.
            file_path (str): Path to the file being analysed.

        Returns:
            Any: The computed metric value (typically int or float).
        """
        raise NotImplementedError("Plugin must implement extract()")

    def confidence_score(self, pyflakes_output: List[str]) -> float:
        """
        Optionally return a confidence score (0.0–1.0) for this metric.

        Args:
            pyflakes_output (List[str]): Pyflakes diagnostics.

        Returns:
            float: Confidence in the metric's accuracy.
        """
        return 1.0

    def severity_level(self, pyflakes_output: List[str]) -> str:
        """
        Optionally classify this metric’s result severity.

        Args:
            pyflakes_output (List[str]): Pyflakes diagnostics.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        return "low"
