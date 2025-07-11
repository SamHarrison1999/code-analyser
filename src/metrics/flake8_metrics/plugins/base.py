"""
Base class for Flake8 metric plugins.

Each plugin must define:
- name(): a unique metric name
- extract(): a method to compute the metric from parsed Flake8 output
"""

from abc import ABC, abstractmethod
from typing import Any, List


class Flake8Plugin(ABC):
    """
    Abstract base class for Flake8 metric plugins.

    All Flake8 plugins must:
    - Provide a unique metric name via `name()`
    - Implement `extract()` to compute a metric from Flake8 diagnostics
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Returns:
            str: The unique name of the plugin metric (used as a key in output dictionaries).
        """
        pass

    @abstractmethod
    def extract(self, flake8_output: List[str], file_path: str) -> Any:
        """
        Computes a metric from the Flake8 output lines.

        Args:
            flake8_output (List[str]): Parsed Flake8 diagnostic lines.
            file_path (str): Path to the source file being analysed.

        Returns:
            Any: The computed metric value (e.g., int or float).
        """
        pass
