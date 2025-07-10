"""
Base class for Pydocstyle metric plugins.

Each plugin inspects the Pydocstyle output and extracts one metric.
"""

from abc import ABC, abstractmethod
from typing import Any


class PydocstylePlugin(ABC):
    """
    Abstract base class for Pydocstyle metric plugins.
    Plugins operate on parsed Pydocstyle diagnostic lines.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Returns the unique name of the plugin metric.
        """
        pass

    @abstractmethod
    def extract(self, pydocstyle_output: list[str], file_path: str) -> Any:
        """
        Computes a metric from Pydocstyle output.

        Args:
            pydocstyle_output (list[str]): Raw lines of Pydocstyle output.
            file_path (str): Path to the source file being analysed.

        Returns:
            Any: The computed metric (int, float, etc.).
        """
        pass
