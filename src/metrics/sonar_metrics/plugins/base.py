# File: code_analyser/src/metrics/sonar_metrics/plugins/base.py

"""
Base class for SonarQube metric plugins.

All plugins must implement:
- name(): returns a unique metric name used for indexing/export
- extract(sonar_data, file_path): computes the metric value from SonarQube results

This base class standardises plugin interfaces across the Sonar metrics system.
"""

from abc import ABC, abstractmethod
from typing import Any, Union


class SonarMetricPlugin(ABC):
    """
    Abstract base class for all SonarQube metric plugins.

    Each plugin extracts a specific metric from SonarQube's analysis results.
    """

    # ✅ Best Practice: Allow plugins to carry metadata
    plugin_name: str = ""
    plugin_tags: list[str] = []

    @abstractmethod
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        This name will be used as a dictionary key in structured outputs.

        Returns:
            str: Metric identifier (e.g., 'bugs', 'vulnerabilities').
        """
        raise NotImplementedError("Subclasses must implement the name() method.")

    @abstractmethod
    def extract(self, sonar_data: dict[str, Any], file_path: str) -> Union[int, float]:
        """
        Computes the metric value using parsed SonarQube analysis data.

        Args:
            sonar_data (dict[str, Any]): Parsed SonarQube results.
            file_path (str): Path to the analysed file (may be ignored if unused).

        Returns:
            Union[int, float]: The computed metric value.
        """
        raise NotImplementedError("Subclasses must implement the extract() method.")

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Optionally return a confidence score (0.0–1.0) for this metric.

        Returns:
            float: Confidence in the metric's accuracy.
        """
        return 1.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Optionally classify the severity of this metric's result.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        return "low"
