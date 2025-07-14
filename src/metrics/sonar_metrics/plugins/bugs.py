from .base import SonarMetricPlugin
from typing import Any
import logging


class BugsPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'bugs' metric from SonarQube analysis data.
    """

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return "bugs"  # ✅ Best Practice: Prefix with 'sonar.' to distinguish source

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
            logging.warning(f"[BugsPlugin] Failed to extract 'bugs' metric for {file_path}: {e}")
            return 0.0
