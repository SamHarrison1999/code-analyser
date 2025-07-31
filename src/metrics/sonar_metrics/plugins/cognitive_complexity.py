# File: code_analyser/src/metrics/sonar_metrics/plugins/cognitive_complexity.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class CognitiveComplexityPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'cognitive_complexity' metric from SonarQube analysis data.
    """

    # ✅ Metadata for filtering, dashboards, and plugin discovery
    plugin_name = "cognitive_complexity"
    plugin_tags = ["complexity", "sonarqube", "maintainability"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'cognitive_complexity' metric from the SonarQube results.

        Args:
            sonar_data (dict[str, Any]): Parsed SonarQube API results.
            file_path (str): Path to the analysed file (not used by this plugin).

        Returns:
            float: The cognitive complexity score or 0.0 if unavailable or invalid.
        """
        try:
            value = sonar_data.get("cognitive_complexity", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[CognitiveComplexityPlugin] Failed to extract metric for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Confidence score based on presence of the cognitive complexity field.

        Returns:
            float: Confidence score (1.0 = present, 0.0 = absent).
        """
        return 1.0 if "cognitive_complexity" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classify the cognitive complexity severity based on threshold.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            complexity = float(sonar_data.get("cognitive_complexity", 0))
            if complexity <= 5:
                return "low"
            elif complexity <= 15:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
