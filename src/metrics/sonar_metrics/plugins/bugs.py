# File: code_analyser/src/metrics/sonar_metrics/plugins/bugs.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class BugsPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'bugs' metric from SonarQube analysis data.
    """

    # ✅ Best Practice: Define metadata for plugin discovery and filtering
    plugin_name = "bugs"
    plugin_tags = ["issues", "defects", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'bugs' metric from the raw SonarQube analysis data.

        Args:
            sonar_data (dict[str, Any]): The full dictionary of SonarQube results.
            file_path (str): The path to the analysed file (not used here).

        Returns:
            float: The number of bugs found, or 0.0 if unavailable or invalid.
        """
        try:
            value = sonar_data.get("bugs", 0)
            return float(value)
        except (TypeError, ValueError) as e:
            logging.warning(
                f"[BugsPlugin] Failed to extract 'bugs' metric for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns a confidence score based on data presence.

        Returns:
            float: Confidence in the metric's presence.
        """
        return 1.0 if "bugs" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies the severity based on the number of bugs.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            bugs = float(sonar_data.get("bugs", 0))
            if bugs == 0:
                return "low"
            elif bugs <= 5:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
