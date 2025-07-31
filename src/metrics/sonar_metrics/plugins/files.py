# File: code_analyser/src/metrics/sonar_metrics/plugins/files.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class FilesPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'files' metric from SonarQube analysis data.
    """

    # ✅ Plugin metadata for plugin registry and UI filtering
    plugin_name = "files"
    plugin_tags = ["structure", "metadata", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'files' metric from the SonarQube analysis data.

        Args:
            sonar_data (dict[str, Any]): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (unused).

        Returns:
            float: The number of files analysed, or 0.0 if missing or invalid.
        """
        try:
            value = sonar_data.get("files", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[FilesPlugin] Failed to extract 'files' metric for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on availability of the 'files' key.

        Returns:
            float: 1.0 if present, 0.0 otherwise.
        """
        return 1.0 if "files" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Assigns severity based on the number of files.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            count = float(sonar_data.get("files", 0))
            if count <= 1:
                return "low"
            elif count <= 20:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
