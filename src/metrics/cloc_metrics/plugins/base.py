"""
Base class for CLOC metric plugins.

All plugins must implement:
- .name(): returns a unique metric name
- .extract(cloc_data): returns the computed value from CLOC JSON data
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseClocMetricPlugin(ABC):
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the metric this plugin computes.
        """
        pass

    @abstractmethod
    def extract(self, cloc_data: dict[str, Any]) -> Any:
        """
        Compute and return the metric value based on the cloc JSON structure.
        """
        pass
