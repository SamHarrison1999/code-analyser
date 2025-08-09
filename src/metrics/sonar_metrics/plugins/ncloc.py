# File: code_analyser/src/metrics/sonar_metrics/plugins/ncloc.py

from .base import SonarMetricPlugin
from typing import Any
import logging


class NclocPlugin(SonarMetricPlugin):
    """
    Plugin to extract the 'ncloc' (non-comment lines of code) metric from SonarQube data.
    """

    # ✅ Metadata for plugin registry and filtering
    plugin_name = "ncloc"
    plugin_tags = ["size", "lines", "sonarqube"]

    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return self.plugin_name

    def extract(self, sonar_data: dict[str, Any], file_path: str) -> float:
        """
        Extracts the 'ncloc' (non-comment lines of code) metric from the SonarQube data.

        Args:
            sonar_data (dict[str, Any]): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used directly).

        Returns:
            float: The value of ncloc or 0.0 if unavailable or invalid.
        """
        try:
            value = sonar_data.get("ncloc", 0)
            return float(value)
        except (ValueError, TypeError) as e:
            logging.warning(f"[NclocPlugin] Failed to extract 'ncloc' for {file_path}: {e}")
            return 0.0

    def confidence_score(self, sonar_data: dict[str, Any]) -> float:
        """
        Confidence based on presence of 'ncloc'.

        Returns:
            float: 1.0 if present, 0.0 otherwise.
        """
        return 1.0 if "ncloc" in sonar_data else 0.0

    def severity_level(self, sonar_data: dict[str, Any]) -> str:
        """
        Classifies code size severity based on non-comment lines of code.

        Returns:
            str: One of 'low', 'medium', or 'high'.
        """
        try:
            ncloc = float(sonar_data.get("ncloc", 0))
            if ncloc <= 100:
                return "low"
            elif ncloc <= 500:
                return "medium"
            else:
                return "high"
        except Exception:
            return "low"
