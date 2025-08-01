# File: code_analyser/src/metrics/sonar_metrics/plugins/success_density.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class TestSuccessDensityPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'test_success_density' metric from SonarQube analysis results.
    """

    # ✅ Plugin metadata for structured plugin loading and GUI filtering
    plugin_name = "test_success_density"
    plugin_tags = ["testing", "coverage", "quality", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'test_success_density' metric from SonarQube analysis results.

        Args:
            sonar_data (dict[str, Any]): Parsed dictionary from SonarQube metrics API.
            file_path (str): Path to the analysed file (not used by this plugin).

        Returns:
            float: The success density of tests, or 0.0 if unavailable or invalid.
        """
        try:
            value = sonar_data.get("test_success_density", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[TestSuccessDensityPlugin] Failed to extract 'test_success_density' for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on whether the value is present.

        Returns:
            float: 1.0 if present, 0.0 otherwise.
        """
        return 1.0 if "test_success_density" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies severity based on test success percentage.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            success_rate = float(sonar_data.get("test_success_density", 0))
            if success_rate >= 90.0:
                return "low"
            elif success_rate >= 60.0:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
