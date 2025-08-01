# File: code_analyser/src/metrics/sonar_metrics/plugins/number_of_tests.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class TestsPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'tests' metric from SonarQube analysis data.
    """

    # ✅ Plugin metadata for structured filtering and dashboarding
    plugin_name = "tests"
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
        Extracts the 'tests' metric from the SonarQube analysis data.

        Args:
            sonar_data (dict[str, Any]): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used directly).

        Returns:
            float: The number of tests reported, or 0.0 if missing or invalid.
        """
        try:
            value = sonar_data.get("tests", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[TestsPlugin] Failed to extract 'tests' metric for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Returns confidence score based on data presence.

        Returns:
            float: 1.0 if 'tests' is present, otherwise 0.0.
        """
        return 1.0 if "tests" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies severity based on number of tests present.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            test_count = float(sonar_data.get("tests", 0))
            if test_count == 0:
                return "high"
            elif test_count <= 5:
                return "medium"
            else:
                return "low"
        except Exception:
            return "medium"
