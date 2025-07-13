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

    @staticmethod
    @abstractmethod
    def name() -> str:
        """
        Return the unique name of the metric this plugin computes.
        """
        pass

    @abstractmethod
    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> Any:
        """
        Compute and return the metric value from the parsed Pylint output.
        """
        pass
