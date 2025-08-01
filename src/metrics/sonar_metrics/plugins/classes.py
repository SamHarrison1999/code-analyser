# File: code_analyser/src/metrics/sonar_metrics/plugins/classes.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class ClassesPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'classes' metric from SonarQube analysis data.
    """

    # ✅ Best Practice: Include metadata for discovery, filtering, and GUI tagging
    plugin_name = "classes"
    plugin_tags = ["structure", "object-oriented", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'classes' metric from the raw SonarQube analysis data.

        Args:
            sonar_data (dict[str, Any]): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used by this plugin).

        Returns:
            float: The number of classes reported by SonarQube, or 0.0 if missing or invalid.
        """
        try:
            value = sonar_data.get("classes", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[ClassesPlugin] Failed to extract 'classes' for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Confidence score based on data availability.

        Returns:
            float: 1.0 if value present, else 0.0
        """
        return 1.0 if "classes" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies severity based on number of classes.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            classes = float(sonar_data.get("classes", 0))
            if classes <= 1:
                return "low"
            elif classes <= 10:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
