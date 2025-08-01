# File: code_analyser/src/metrics/sonar_metrics/plugins/duplicated_blocks.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class DuplicatedBlocksPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'duplicated_blocks' metric from SonarQube analysis data.
    """

    # ✅ Plugin metadata for discovery, filtering, and classification
    plugin_name = "duplicated_blocks"
    plugin_tags = ["duplication", "code_smell", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique metric name provided by this plugin.

        Returns:
            str: Metric name used in the result dictionary.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'duplicated_blocks' metric from SonarQube results.

        Args:
            sonar_data (dict[str, Any]): SonarQube API data for the analysed file.
            file_path (str): Path to the target file (unused).

        Returns:
            float: Number of duplicated blocks, or 0.0 if unavailable or invalid.
        """
        try:
            value = sonar_data.get("duplicated_blocks", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[DuplicatedBlocksPlugin] Failed to extract for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on data presence.

        Returns:
            float: Confidence score (1.0 if present, else 0.0).
        """
        return 1.0 if "duplicated_blocks" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies severity based on the number of duplicated blocks.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            blocks = float(sonar_data.get("duplicated_blocks", 0))
            if blocks <= 1:
                return "low"
            elif blocks <= 5:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
