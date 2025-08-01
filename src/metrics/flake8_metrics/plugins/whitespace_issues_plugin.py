# File: code_analyser/src/metrics/flake8_metrics/plugins/whitespace_issues_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class WhitespaceIssuesPlugin(Flake8MetricPlugin):
    """
    Counts the number of Flake8 whitespace-related issues,
    typically indicated by E2xx and W2xx codes.
    """

    # ✅ Best Practice: Register plugin metadata for registry and overlay filtering
    plugin_name = "number_of_whitespace_issues"
    plugin_tags = ["whitespace", "E2", "W2", "style", "flake8"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Assumes Flake8 output is properly formed and uses standard E/W codes
    # 🧠 ML Signal: Whitespace consistency reflects formatting habits and tool adherence
    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Count the number of whitespace-related issues in Flake8 output.

        Args:
            flake8_output (List[str]): Parsed Flake8 output lines.
            file_path (str): Path to the analysed file (not used directly).

        Returns:
            int: Count of lines reporting whitespace issues (typically E2xx or W2xx codes).
        """
        return sum(
            1
            for line in flake8_output
            if any(code in line for code in [": E2", ": W2"])
        )

    # ✅ Best Practice: High confidence when flake8_output is present
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0 if flake8_output else 0.0

    # ✅ Best Practice: Severity based on violation volume
    def severity_level(self, flake8_output: List[str]) -> str:
        count = self.extract(flake8_output, file_path="")
        if count >= 10:
            return "high"
        elif count >= 3:
            return "medium"
        return "low"
