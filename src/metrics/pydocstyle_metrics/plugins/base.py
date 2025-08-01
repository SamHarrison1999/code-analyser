# File: code_analyser/src/metrics/pydocstyle_metrics/plugins/base.py

"""
Base class for Pydocstyle metric plugins.

Each plugin inspects parsed Pydocstyle output and extracts a specific scalar metric.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class PydocstyleMetricPlugin(ABC):
    """
    Abstract base class for Pydocstyle metric plugins.

    Plugins must:
    - Define a unique metric name
    - Extract a scalar value (typically int or float) from Pydocstyle diagnostics
    - Optionally provide confidence and severity annotations
    """

    # ✅ Best Practice: Plugin metadata for filtering and discovery
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
    def extract(self, pydocstyle_output: List[str], file_path: str) -> Any:
        """
        Compute the metric from the provided Pydocstyle diagnostic output.

        Args:
            pydocstyle_output (List[str]): Raw Pydocstyle output lines.
            file_path (str): Path to the file being analysed.

        Returns:
            Any: The computed metric value (typically int or float).
        """
        raise NotImplementedError("Plugin must implement extract()")

    def confidence_score(self, pydocstyle_output: List[str]) -> float:
        """
        Optionally return a confidence score (0.0–1.0) for this metric.

        Returns:
            float: Confidence in the metric's accuracy.
        """
        return 1.0

    def severity_level(self, pydocstyle_output: List[str]) -> str:
        """
        Optionally classify this metric’s result severity.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        return "low"
