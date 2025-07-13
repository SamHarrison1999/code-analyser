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

    Each plugin must:
    - Provide a human-readable metric name
    - Implement a method to extract the metric value from unused Vulture items
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: Unique metric identifier (e.g., 'unused_functions')
        """
        raise NotImplementedError("Plugin must implement the name() method.")

    @abstractmethod
    def extract(self, vulture_items: list[Any]) -> Union[int, float]:
        """
        Computes the metric value from a list of Vulture items.

        Args:
            vulture_items (list[Any]): List of Vulture results (with attributes like 'typ', 'name', 'lineno', etc.)

        Returns:
            int | float: Computed metric result (e.g., count, percentage)
        """
        raise NotImplementedError("Plugin must implement the extract() method.")
