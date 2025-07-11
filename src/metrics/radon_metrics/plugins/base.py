"""
Base class for Radon metric plugins.

All plugins must implement:
- name(): returns a unique metric name used for indexing/export
- extract(radon_data): computes the metric value from parsed Radon JSON

This base class standardises plugin interfaces across the metrics system.
"""

from abc import ABC, abstractmethod
from typing import Any


class RadonMetricPlugin(ABC):
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the metric this plugin computes.

        Returns:
            str: Unique metric identifier (e.g. 'number_of_logical_lines')
        """
        pass

    @abstractmethod
    def extract(self, radon_data: dict[str, Any]) -> float | int:
        """
        Compute and return the metric value based on the Radon JSON structure.

        Args:
            radon_data (dict): Parsed Radon output.

        Returns:
            float | int: The computed metric value.
        """
        pass
