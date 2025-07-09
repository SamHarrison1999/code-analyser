"""
Base class for Lizard metric plugins.

Each plugin inspects the Lizard output and extracts one metric.
"""

from abc import ABC, abstractmethod
from typing import Any


class LizardPlugin(ABC):
    """
    Abstract base class for Lizard metric plugins.
    Plugins operate on parsed Lizard diagnostic lines.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Returns the unique name of the plugin metric.
        """
        pass

    @abstractmethod
    def extract(self, lizard_output: list[str], file_path: str) -> Any:
        """
        Computes a metric from Lizard output.

        Args:
            lizard_output (list[str]): Raw lines of Lizard output.
            file_path (str): Path to the source file being analysed.

        Returns:
            Any: The computed metric (int, float, etc.).
        """
        pass
