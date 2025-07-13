"""
Base class for Pydocstyle metric plugins.

Each plugin inspects the Pydocstyle output and extracts one metric.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class PydocstyleMetricPlugin(ABC):
    """
    Abstract base class for Pydocstyle metric plugins.

    Plugins must:
    - Define a unique metric name
    - Implement an extraction method using parsed Pydocstyle output lines
    """

    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of the metric provided by this plugin.

        Returns:
            str: Metric name used as a dictionary key.
        """
        raise NotImplementedError("Plugin must implement name()")

    @abstractmethod
    def extract(self, pydocstyle_output: List[str], file_path: str) -> Any:
        """
        Compute the metric from the provided Pydocstyle diagnostic output.

        Args:
            pydocstyle_output (List[str]): Raw Pydocstyle output lines.
            file_path (str): Path to the file being analysed.

        Returns:
            Any: The computed metric value (typically int or float).
        """
        raise NotImplementedError("Plugin must implement extract()")
