# File: code_analyser/src/metrics/vulture_metrics/plugins/base.py

"""
Base class for Vulture metric plugins.

All plugins must implement:
- name(): returns a unique metric name used for indexing/export
- extract(vulture_items): computes the metric value from raw Vulture unused items

This base class standardises plugin interfaces across the metrics system.
"""

from abc import ABC, abstractmethod
from typing import Any, Union


class VultureMetricPlugin(ABC):
    """
    Abstract base class for all Vulture metric plugins.

    Each plugin extracts a specific metric from a list of Vulture's unused code items.
    """

    # ✅ Plugin metadata for discovery and filtering
    plugin_name: str = ""
    plugin_tags: list[str] = []

    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: Unique metric identifier (e.g., 'unused_functions').
        """
        raise NotImplementedError("Plugin must implement the name() method.")

    @abstractmethod
    def extract(self, vulture_items: list[Any]) -> Union[int, float]:
        """
        Computes the metric value from a list of Vulture items.

        Args:
            vulture_items (list[Any]): List of Vulture result entries (each with 'typ', 'name', 'lineno', etc.).

        Returns:
            int | float: Computed metric result (e.g., count, ratio).
        """
        raise NotImplementedError("Plugin must implement the extract() method.")

    def confidence_score(self, vulture_items: list[Any]) -> float:
        """
        Optionally returns a confidence score for this metric.

        Args:
            vulture_items (list[Any]): The full list of parsed Vulture results.

        Returns:
            float: Confidence score from 0.0 to 1.0. Default is 1.0 (high confidence).
        """
        return 1.0

    def severity_level(self, vulture_items: list[Any]) -> str:
        """
        Optionally classifies severity based on extracted results.

        Args:
            vulture_items (list[Any]): The full list of Vulture results.

        Returns:
            str: One of 'low', 'medium', or 'high'. Default is 'low'.
        """
        return "low"
