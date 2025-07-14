from .base import SonarMetricPlugin
from typing import Any


class ClassesPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return "classes"

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'classes' metric from the raw SonarQube analysis data.

        Args:
            sonar_data (dict[str, Any]): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used by this plugin).

        Returns:
            float: The number of classes reported by SonarQube, or 0.0 if missing or invalid.
        """
        value = sonar_data.get("classes", 0)
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
