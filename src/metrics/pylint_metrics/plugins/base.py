"""
Base class for Pylint metric plugins.

Each plugin inspects the Pylint output and extracts one metric.
"""

from abc import ABC, abstractmethod
from typing import Any


class PylintMetricPlugin(ABC):
    """
    Abstract base class for Pylint metric plugins.
    Plugins operate on parsed Pylint diagnostics and compute a single metric.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Returns the unique name of the plugin metric.
        """
        pass

    @abstractmethod
    def extract(self, pylint_output: list[dict], file_path: str) -> Any:
        """
        Computes a metric from Pylint output.

        Args:
            pylint_output (list[dict]): Parsed Pylint diagnostic messages.
            file_path (str): Path to the source file being analysed.

        Returns:
            Any: The computed metric (e.g., int, float).
        """
        pass
