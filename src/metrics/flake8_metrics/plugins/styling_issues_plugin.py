# File: code_analyser/src/metrics/flake8_metrics/plugins/styling_issues_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class TotalIssuePlugin(Flake8MetricPlugin):
    """
    Counts the total number of Flake8 diagnostic issues reported,
    regardless of code prefix (E, W, F, C, D, N, etc.).
    This serves as an overall summary metric for Flake8.
    """

    # ✅ Best Practice: Unified naming for global issue metric
    plugin_name = "number_of_styling_issues"
    plugin_tags = ["flake8", "total", "summary", "all"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Blind count assumes Flake8 emits only parseable lines
    # 🧠 ML Signal: Total issue count serves as a global proxy for file hygiene
    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts the total number of Flake8 diagnostic issues reported.

        Args:
            flake8_output (List[str]): List of Flake8 output lines.
            file_path (str): Path to the analysed file (unused in this plugin).

        Returns:
            int: Total number of Flake8-reported issues.
        """
        return len(flake8_output)

    # ✅ Best Practice: Confidence based on Flake8 output presence
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0 if flake8_output else 0.0

    # ✅ Best Practice: Severity scaled to total issue volume
    def severity_level(self, flake8_output: List[str]) -> str:
        count = self.extract(flake8_output, file_path="")
        if count >= 30:
            return "high"
        elif count >= 10:
            return "medium"
        return "low"
