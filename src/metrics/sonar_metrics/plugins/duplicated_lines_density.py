# File: code_analyser/src/metrics/sonar_metrics/plugins/duplicated_lines_density.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class DuplicatedLinesDensityPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'duplicated_lines_density' metric from SonarQube analysis data.
    """

    # ✅ Plugin metadata for structured discovery and GUI/ML filtering
    plugin_name = "duplicated_lines_density"
    plugin_tags = ["duplication", "density", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique metric name provided by this plugin.

        Returns:
            str: Metric name used in the result dictionary.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'duplicated_lines_density' metric from SonarQube analysis results.

        Args:
            sonar_data (dict[str, Any]): Parsed SonarQube result dictionary.
            file_path (str): Path to the analysed file (not used here).

        Returns:
            float: Percentage of duplicated lines, or 0.0 if missing or invalid.
        """
        try:
            value = sonar_data.get("duplicated_lines_density", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[DuplicatedLinesDensityPlugin] Failed to extract for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on presence of the metric field.

        Returns:
            float: 1.0 if present, otherwise 0.0.
        """
        return 1.0 if "duplicated_lines_density" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies the duplication severity based on percentage density.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            density = float(sonar_data.get("duplicated_lines_density", 0))
            if density <= 2.0:
                return "low"
            elif density <= 10.0:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
