# File: code_analyser/src/metrics/sonar_metrics/plugins/complexity.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class ComplexityPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'complexity' metric from SonarQube analysis data.
    """

    # ✅ Metadata for plugin discovery, filtering, and dashboards
    plugin_name = "complexity"
    plugin_tags = ["complexity", "code_quality", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extract the 'complexity' metric from SonarQube analysis output.

        Args:
            sonar_data (dict[str, Any]): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used by this plugin).

        Returns:
            float: The total code complexity value, or 0.0 if unavailable or invalid.
        """
        try:
            value = sonar_data.get("complexity", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[ComplexityPlugin] Failed to extract 'complexity' for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on data presence.

        Returns:
            float: 1.0 if present, 0.0 if missing.
        """
        return 1.0 if "complexity" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies severity based on total code complexity.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            complexity = float(sonar_data.get("complexity", 0))
            if complexity <= 5:
                return "low"
            elif complexity <= 20:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
