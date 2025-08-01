# File: code_analyser/src/metrics/sonar_metrics/plugins/code_smells.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class CodeSmellsPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'code_smells' metric from SonarQube analysis data.
    """

    # ✅ Best Practice: Provide metadata for plugin discovery and classification
    plugin_name = "code_smells"
    plugin_tags = ["issues", "maintainability", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'code_smells' metric from the raw SonarQube analysis data.

        Args:
            sonar_data (dict[str, Any]): The full dictionary of SonarQube results.
            file_path (str): Path to the analysed file (not used by this plugin).

        Returns:
            float: The number of code smells reported, or 0.0 if missing or invalid.
        """
        try:
            value = sonar_data.get("code_smells", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[CodeSmellsPlugin] Failed to extract 'code_smells' for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on data presence.

        Returns:
            float: 1.0 if data is present, otherwise 0.0.
        """
        return 1.0 if "code_smells" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies the severity based on the number of code smells.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            smells = float(sonar_data.get("code_smells", 0))
            if smells == 0:
                return "low"
            elif smells <= 10:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
