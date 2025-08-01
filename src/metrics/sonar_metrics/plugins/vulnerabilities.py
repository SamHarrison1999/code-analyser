# File: code_analyser/src/metrics/sonar_metrics/plugins/vulnerabilities.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class VulnerabilitiesPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'vulnerabilities' metric from SonarQube analysis results.
    """

    # ✅ Metadata for structured discovery and classification
    plugin_name = "vulnerabilities"
    plugin_tags = ["security", "issues", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'vulnerabilities' metric from SonarQube analysis results.

        Args:
            sonar_data (dict[str, Any]): Parsed dictionary from SonarQube API.
            file_path (str): Path to the analysed file (unused).

        Returns:
            float: The number of reported vulnerabilities, or 0.0 if unavailable.
        """
        try:
            value = sonar_data.get("vulnerabilities", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[VulnerabilitiesPlugin] Failed to extract 'vulnerabilities' for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on field presence.

        Returns:
            float: 1.0 if present, else 0.0.
        """
        return 1.0 if "vulnerabilities" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies severity based on the number of vulnerabilities.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            count = float(sonar_data.get("vulnerabilities", 0))
            if count == 0:
                return "low"
            elif count <= 5:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
