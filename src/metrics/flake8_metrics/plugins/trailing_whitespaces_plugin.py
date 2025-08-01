# File: code_analyser/src/metrics/flake8_metrics/plugins/trailing_whitespaces_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class TrailingWhitespacesPlugin(Flake8MetricPlugin):
    """
    Counts the number of lines with trailing whitespace issues,
    typically reported by Flake8 with E2xx or W291 codes.
    """

    # ✅ Best Practice: Plugin metadata for filtering and discovery
    plugin_name = "number_of_trailing_whitespaces"
    plugin_tags = ["style", "whitespace", "E2", "W291", "flake8"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Assumes Flake8 output is line-based and well-formed with known codes
    # 🧠 ML Signal: Trailing whitespace is often associated with formatting sloppiness or missing pre-commit hooks
    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts the number of lines with trailing whitespace errors in Flake8 output.

        Args:
            flake8_output (List[str]): List of Flake8 diagnostic strings.
            file_path (str): Path to the file being analysed (unused here).

        Returns:
            int: Count of trailing whitespace issues (typically E2xx).
        """
        return sum(1 for line in flake8_output if ": W291" in line or ": E2" in line)

    # ✅ Best Practice: Confidence is based on presence of Flake8 output
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0 if flake8_output else 0.0

    # ✅ Best Practice: Severity depends on number of whitespace violations
    def severity_level(self, flake8_output: List[str]) -> str:
        count = self.extract(flake8_output, file_path="")
        if count >= 10:
            return "high"
        elif count >= 3:
            return "medium"
        return "low"
