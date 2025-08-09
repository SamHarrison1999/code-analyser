# File: code_analyser/src/metrics/sonar_metrics/plugins/duplicated_lines.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class DuplicatedLinesPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'duplicated_lines' metric from SonarQube analysis data.
    """

    # ✅ Metadata for plugin auto-discovery and filtering
    plugin_name = "duplicated_lines"
    plugin_tags = ["duplication", "technical_debt", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique metric name provided by this plugin.

        Returns:
            str: Metric name used in the result dictionary.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'duplicated_lines' metric from SonarQube results.

        Args:
            sonar_data (dict[str, Any]): Parsed SonarQube analysis result.
            file_path (str): Path to the analysed file (unused in this plugin).

        Returns:
            float: Number of duplicated lines, or 0.0 if missing or invalid.
        """
        try:
            value = sonar_data.get("duplicated_lines", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(f"[DuplicatedLinesPlugin] Failed to extract for {file_path}: {e}")
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on data availability.

        Returns:
            float: 1.0 if 'duplicated_lines' present, else 0.0.
        """
        return 1.0 if "duplicated_lines" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies severity based on number of duplicated lines.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            lines = float(sonar_data.get("duplicated_lines", 0))
            if lines <= 5:
                return "low"
            elif lines <= 20:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
