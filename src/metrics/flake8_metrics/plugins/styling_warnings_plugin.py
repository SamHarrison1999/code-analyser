# File: code_analyser/src/metrics/flake8_metrics/plugins/styling_warnings_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class StylingWarningPlugin(Flake8MetricPlugin):
    """
    Counts the number of Flake8 style warnings related to codes starting with 'W'
    (e.g., whitespace issues, line spacing, and minor formatting).
    """

    # ✅ Best Practice: Plugin metadata for dynamic discovery and tag-based filtering
    plugin_name = "number_of_styling_warnings"
    plugin_tags = ["style", "warning", "W", "flake8", "formatting"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Depends on Flake8 output format and correct prefix tagging ('W')
    # 🧠 ML Signal: Style warning frequency may signal inconsistent formatting habits
    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts the number of Flake8 warnings related to styling (e.g., 'W' codes).
        """
        return sum(1 for line in flake8_output if ": W" in line or " W" in line)

    # ✅ Best Practice: Confidence based on Flake8 output visibility
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0 if flake8_output else 0.0

    # ✅ Best Practice: Assign severity based on volume of W-prefixed violations
    def severity_level(self, flake8_output: List[str]) -> str:
        count = self.extract(flake8_output, file_path="")
        if count >= 15:
            return "high"
        elif count >= 5:
            return "medium"
        return "low"
