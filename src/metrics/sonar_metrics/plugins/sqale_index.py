# File: code_analyser/src/metrics/sonar_metrics/plugins/sqale_index.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class SqaleIndexPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'sqale_index' (technical debt cost in minutes) from SonarQube analysis data.
    """

    # ✅ Metadata for plugin discovery and filtering
    plugin_name = "sqale_index"
    plugin_tags = ["technical_debt", "maintainability", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'sqale_index' metric from the SonarQube analysis data.

        Args:
            sonar_data (dict[str, Any]): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used directly).

        Returns:
            float: The SQALE index value (minutes of technical debt), or 0.0 if missing or invalid.
        """
        try:
            value = sonar_data.get("sqale_index", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[SqaleIndexPlugin] Failed to extract 'sqale_index' for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on presence of the metric.

        Returns:
            float: 1.0 if present, otherwise 0.0.
        """
        return 1.0 if "sqale_index" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies severity based on SQALE index value.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            debt = float(sonar_data.get("sqale_index", 0))
            if debt <= 30:
                return "low"
            elif debt <= 300:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
