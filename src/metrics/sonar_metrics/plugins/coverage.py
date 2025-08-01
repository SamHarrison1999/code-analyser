# File: code_analyser/src/metrics/sonar_metrics/plugins/coverage.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class CoveragePlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'coverage' metric from SonarQube analysis data.
    """

    # ✅ Plugin metadata for dynamic filtering and reporting
    plugin_name = "coverage"
    plugin_tags = ["testing", "coverage", "quality", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'coverage' metric from SonarQube analysis output.

        Args:
            sonar_data (dict[str, Any]): Dictionary of parsed SonarQube results.
            file_path (str): Path to the analysed file (unused).

        Returns:
            float: The code coverage percentage, or 0.0 if unavailable or invalid.
        """
        try:
            value = sonar_data.get("coverage", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[CoveragePlugin] Failed to extract 'coverage' for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Confidence score based on availability of the 'coverage' key.

        Returns:
            float: Confidence score (1.0 if present, 0.0 if not).
        """
        return 1.0 if "coverage" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classify severity based on test coverage thresholds.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            coverage = float(sonar_data.get("coverage", 0))
            if coverage >= 80.0:
                return "low"
            elif coverage >= 50.0:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
