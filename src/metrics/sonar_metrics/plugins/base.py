"""
Base class for SonarQube metric plugins.

All plugins must implement:
- name(): returns a unique metric name used for indexing/export
- extract(sonar_data, file_path): computes the metric value from SonarQube results
"""

from abc import ABC, abstractmethod
from typing import Any, Union


class SonarMetricPlugin(ABC):
    """
    Abstract base class for all SonarQube metric plugins.

    Each plugin extracts a specific metric from SonarQube's analysis results.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: Metric identifier (e.g., 'coverage', 'code_smells').
        """
        raise NotImplementedError("Subclasses must implement the name() method.")

    @abstractmethod
    def extract(self, sonar_data: dict[str, Any], file_path: str) -> Union[int, float]:
        """
        Computes the metric value using parsed SonarQube analysis data.

        Args:
            sonar_data (dict[str, Any]): Parsed SonarQube results.
            file_path (str): Path to the analysed file (can be ignored if unused).

        Returns:
            Union[int, float]: The computed metric value for this metric.
        """
        raise NotImplementedError("Subclasses must implement the extract() method.")
