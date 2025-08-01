# File: code_analyser/src/metrics/flake8_metrics/plugins/unused_imports_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class UnusedImportPlugin(Flake8MetricPlugin):
    """
    Counts the number of unused import warnings (F401) reported by Flake8.
    This typically indicates dead code or unnecessary dependencies.
    """

    # ✅ Best Practice: Plugin metadata for dynamic discovery and filtering
    plugin_name = "number_of_unused_imports"
    plugin_tags = ["F401", "imports", "cleanup", "dead_code", "flake8"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: This check assumes Flake8 output format is consistent with F401 codes
    # 🧠 ML Signal: Unused imports correlate with lack of refactoring or excessive coupling
    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Count the number of unused import warnings (F401) reported by Flake8.

        Args:
            flake8_output (List[str]): Parsed Flake8 output lines.
            file_path (str): Path to the analysed file (not used here).

        Returns:
            int: Number of F401 unused import violations.
        """
        return sum(1 for line in flake8_output if ": F401 " in line)

    # ✅ Best Practice: Full confidence when flake8 runs successfully
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0 if flake8_output else 0.0

    # ✅ Best Practice: Severity increases with number of F401 occurrences
    def severity_level(self, flake8_output: List[str]) -> str:
        count = self.extract(flake8_output, file_path="")
        if count >= 10:
            return "high"
        elif count >= 3:
            return "medium"
        return "low"
