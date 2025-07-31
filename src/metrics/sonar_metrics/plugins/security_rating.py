# File: code_analyser/src/metrics/sonar_metrics/plugins/security_rating.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class SecurityRatingPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'security_rating' metric from SonarQube analysis data.
    """

    # ✅ Plugin metadata for discovery and classification
    plugin_name = "security_rating"
    plugin_tags = ["security", "rating", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'security_rating' metric from the SonarQube analysis data.

        Args:
            sonar_data (dict[str, Any]): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used directly).

        Returns:
            float: The security rating, or 0.0 if missing or invalid.
        """
        try:
            value = sonar_data.get("security_rating", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[SecurityRatingPlugin] Failed to extract 'security_rating' for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on data presence.

        Returns:
            float: 1.0 if present, 0.0 otherwise.
        """
        return 1.0 if "security_rating" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies severity based on numeric security rating (1 = best, 5 = worst).

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            rating = float(sonar_data.get("security_rating", 0))
            if rating <= 1.0:
                return "low"
            elif rating <= 3.0:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
