"""
Base class for Flake8 metric plugins.

Each plugin inspects the Flake8 output and extracts one metric.
"""

from abc import ABC, abstractmethod
from typing import Any


class Flake8Plugin(ABC):
    """
    Abstract base class for Flake8 metric plugins.
    Plugins operate on parsed Flake8 diagnostic lines.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Returns the unique name of the plugin metric.
        """
        pass

    @abstractmethod
    def extract(self, flake8_output: list[str], file_path: str) -> Any:
        """
        Computes a metric from Flake8 output.

        Args:
            flake8_output (list[str]): Raw lines of Flake8 output.
            file_path (str): Path to the source file being analysed.

        Returns:
            Any: The computed metric (int, float, etc.).
        """
        pass
