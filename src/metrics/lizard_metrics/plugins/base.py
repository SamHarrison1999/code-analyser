"""
Base class for Lizard metric plugins.

Each plugin inspects preprocessed Lizard metric entries and extracts one metric.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict


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
        raise NotImplementedError("LizardPlugin must implement a static name() method.")

    @abstractmethod
    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> Any:
        """
        Computes a metric from Lizard output.

        Args:
            lizard_metrics (List[Dict[str, Any]]): Parsed metric entries with keys like 'name' and 'value'.
            file_path (str): Path to the analysed source file.

        Returns:
            Any: The computed metric (e.g., int, float).
        """
        raise NotImplementedError("LizardPlugin must implement the extract() method.")
