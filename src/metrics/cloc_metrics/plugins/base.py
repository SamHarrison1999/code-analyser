"""
Base class for CLOC metric plugins.

All plugins must implement:
- name(): returns a unique metric name used for indexing/export
- extract(cloc_data): computes the metric value from parsed CLOC JSON

This base class standardises plugin interfaces across the metrics system.
"""

from abc import ABC, abstractmethod
from typing import Any


class ClocMetricPlugin(ABC):
    # ✅ Optional metadata for plugin discovery
    plugin_name: str = ""
    plugin_tags: list[str] = []

    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the metric this plugin computes.

        Returns:
            str: Unique metric identifier (e.g. 'total_lines', 'comment_density')
        """
        raise NotImplementedError("ClocMetricPlugin subclasses must implement name().")

    @abstractmethod
    def extract(self, cloc_data: dict[str, Any]) -> float | int:
        """
        Compute and return the metric value based on the CLOC JSON structure.

        Args:
            cloc_data (dict): Parsed CLOC output (usually from cloc --json)

        Returns:
            float | int: The computed metric value
        """
        raise NotImplementedError(
            "ClocMetricPlugin subclasses must implement extract()."
        )

    def confidence_score(self, cloc_data: dict[str, Any]) -> float:
        """
        Optional: Return a confidence level for this metric's accuracy.

        Returns:
            float: Confidence value in range [0.0, 1.0]
        """
        return 1.0

    def severity_level(self, cloc_data: dict[str, Any]) -> str:
        """
        Optional: Provide a severity level ('low', 'medium', 'high') based on metric logic.

        Returns:
            str: Severity label
        """
        return "low"
