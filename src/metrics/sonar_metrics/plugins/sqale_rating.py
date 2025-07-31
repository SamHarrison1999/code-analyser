# File: code_analyser/src/metrics/sonar_metrics/plugins/sqale_rating.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class SqaleRatingPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'sqale_rating' (maintainability rating) from SonarQube analysis results.
    """

    # ✅ Plugin metadata for registry and UI filtering
    plugin_name = "sqale_rating"
    plugin_tags = ["maintainability", "rating", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'sqale_rating' metric from the SonarQube analysis results.

        Args:
            sonar_data (dict[str, Any]): Dictionary returned by the SonarQube API.
            file_path (str): Path to the analysed source file (not used here).

        Returns:
            float: The technical debt rating, or 0.0 if unavailable or invalid.
        """
        try:
            value = sonar_data.get("sqale_rating", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[SqaleRatingPlugin] Failed to extract 'sqale_rating' for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on availability of 'sqale_rating'.

        Returns:
            float: 1.0 if present, else 0.0.
        """
        return 1.0 if "sqale_rating" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies severity based on numeric SQALE rating (1 = best, 5 = worst).

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            rating = float(sonar_data.get("sqale_rating", 0))
            if rating <= 1.0:
                return "low"
            elif rating <= 3.0:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
