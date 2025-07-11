"""
Base class for Pylint metric plugins.

Each plugin inspects the Pylint output and extracts one metric.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict


class PylintMetricPlugin(ABC):
    """
    Abstract base class for Pylint metric plugins.
    Plugins operate on parsed Pylint diagnostics and compute a single metric.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Return the unique name of the metric this plugin provides.

        Returns:
            str: Identifier name used to label the metric.
        """
        pass

    @abstractmethod
    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> Any:
        """
        Compute a single metric from the given Pylint output.

        Args:
            pylint_output (List[Dict[str, Any]]): Parsed Pylint diagnostic messages.
            file_path (str): Absolute or relative path to the analysed Python file.

        Returns:
            Any: The computed metric value (e.g., int, float, or str).
        """
        pass
