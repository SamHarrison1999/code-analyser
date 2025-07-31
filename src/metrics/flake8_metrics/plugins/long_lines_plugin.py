# File: code_analyser/src/metrics/flake8_metrics/plugins/long_lines_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class LongLinePlugin(Flake8MetricPlugin):
    """
    Counts the number of Flake8 violations for long lines (E501).
    These usually indicate code that exceeds the recommended line length.
    """

    # ✅ Best Practice: Register plugin metadata for filtering and GUI display
    plugin_name = "number_of_long_lines"
    plugin_tags = ["E501", "style", "length", "pep8", "formatting"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Relies on Flake8 including E501 output; assumes diagnostics format is stable
    # 🧠 ML Signal: Long line frequency is a proxy for PEP8 compliance and readability
    def extract(self, flake8_output: List[str], file_path: str) -> int:
        return sum(1 for line in flake8_output if " E501" in line)

    # ✅ Best Practice: High confidence when Flake8 results are available
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0 if flake8_output else 0.0

    # ✅ Best Practice: Assign severity based on count of long line violations
    def severity_level(self, flake8_output: List[str]) -> str:
        count = self.extract(flake8_output, file_path="")
        if count >= 15:
            return "high"
        elif count >= 5:
            return "medium"
        return "low"
