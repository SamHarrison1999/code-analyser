# File: code_analyser/src/metrics/flake8_metrics/plugins/inconsistent_indentation_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class InconsistentIndentationPlugin(Flake8MetricPlugin):
    """
    Counts how many Flake8 diagnostics relate to inconsistent indentation,
    specifically targeting codes E111 (indentation is not a multiple of four)
    and E114 (indentation is not a multiple of four — comment line).
    """

    # ✅ Best Practice: Provide unique plugin name and tags
    plugin_name = "number_of_inconsistent_indentations"
    plugin_tags = ["indentation", "style", "E111", "E114", "formatting"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Relies on Flake8 emitting specific codes from pycodestyle
    # 🧠 ML Signal: Indentation inconsistency can suggest poor formatting practices
    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts how many Flake8 diagnostics are due to inconsistent indentation (code E111 or E114).

        Args:
            flake8_output (List[str]): Parsed Flake8 output lines.
            file_path (str): Path to the file being analysed.

        Returns:
            int: Number of inconsistent indentation issues.
        """
        return sum(1 for line in flake8_output if " E111" in line or " E114" in line)

    # ✅ Best Practice: Confidence is high if Flake8 output is present
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0 if flake8_output else 0.0

    # ✅ Best Practice: Severity increases with number of violations
    def severity_level(self, flake8_output: List[str]) -> str:
        count = self.extract(flake8_output, file_path="")
        if count >= 10:
            return "high"
        elif count >= 3:
            return "medium"
        return "low"
