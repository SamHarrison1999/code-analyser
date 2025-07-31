# File: code_analyser/src/metrics/sonar_metrics/plugins/comment_lines_density.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class CommentLinesDensityPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'comment_lines_density' metric from SonarQube analysis data.
    """

    # ✅ Metadata for dynamic discovery, GUI filtering, and ML pipelines
    plugin_name = "comment_lines_density"
    plugin_tags = ["documentation", "comments", "quality", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extract the 'comment_lines_density' metric from SonarQube analysis output.

        Args:
            sonar_data (dict[str, Any]): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used by this plugin).

        Returns:
            float: Percentage of comment lines, or 0.0 if unavailable or invalid.
        """
        try:
            value = sonar_data.get("comment_lines_density", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(
                f"[CommentLinesDensityPlugin] Failed to extract for {file_path}: {e}"
            )
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Confidence score based on presence of the comment_lines_density field.

        Returns:
            float: Confidence score (1.0 = present, 0.0 = absent).
        """
        return 1.0 if "comment_lines_density" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies severity based on percentage of comments.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            density = float(sonar_data.get("comment_lines_density", 0))
            if density >= 25.0:
                return "low"
            elif density >= 10.0:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
