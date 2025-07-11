"""
Base class for Lizard metric plugins.

Each plugin inspects preprocessed Lizard metric entries and extracts one metric.
"""

from abc import ABC, abstractmethod
from typing import Any


class LizardPlugin(ABC):
    """
    Abstract base class for Lizard metric plugins.
    Plugins operate on Lizard's pre-parsed metric entries.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Returns:
            str: The unique name of the plugin metric.
        """
        pass

    @abstractmethod
    def extract(self, lizard_metrics: list[dict], file_path: str) -> Any:
        """
        Computes a metric from Lizard output.

        Args:
            lizard_metrics (list[dict]): List of Lizard metric entries with 'name' and 'value'.
            file_path (str): Path to the analysed source file.

        Returns:
            Any: The computed metric (int, float, etc.).
        """
        pass
