from .base import SonarMetricPlugin
from typing import Any


class CodeSmellsPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return "code_smells"

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'code_smells' metric from the raw SonarQube analysis data.

        Args:
            sonar_data (dict[str, Any]): The full dictionary of SonarQube results.
            file_path (str): Path to the analysed file (not used by this plugin).

        Returns:
            float: The number of code smells reported, or 0.0 if missing or invalid.
        """
        value = sonar_data.get("code_smells")
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
