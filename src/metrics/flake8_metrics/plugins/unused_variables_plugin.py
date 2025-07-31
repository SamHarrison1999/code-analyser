# File: code_analyser/src/metrics/flake8_metrics/plugins/unused_variables_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class UnusedVariablePlugin(Flake8MetricPlugin):
    """
    Counts the number of Flake8 warnings related to unused variables (F841),
    which can indicate sloppy or incomplete logic.
    """

    # ✅ Best Practice: Register metadata for dynamic plugin discovery
    plugin_name = "number_of_unused_variables"
    plugin_tags = ["F841", "unused", "variables", "cleanup", "flake8"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Assumes diagnostics follow the standard ': F841' Flake8 pattern
    # 🧠 ML Signal: Unused variables may correlate with incomplete logic or developer error
    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Count the number of unused variable warnings (F841) reported by Flake8.

        Args:
            flake8_output (List[str]): Parsed Flake8 output lines.
            file_path (str): Path to the analysed file (not used in logic).

        Returns:
            int: Count of F841 violations for unused variables.
        """
        return sum(1 for line in flake8_output if ": F841 " in line)

    # ✅ Best Practice: Fully confident if Flake8 diagnostics are present
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0 if flake8_output else 0.0

    # ✅ Best Practice: Classify severity based on volume of violations
    def severity_level(self, flake8_output: List[str]) -> str:
        count = self.extract(flake8_output, file_path="")
        if count >= 5:
            return "high"
        elif count >= 2:
            return "medium"
        return "low"
