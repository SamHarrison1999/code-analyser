# File: code_analyser/src/metrics/flake8_metrics/plugins/naming_issues_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class NamingIssuePlugin(Flake8MetricPlugin):
    """
    Counts the number of naming convention issues reported by Flake8,
    typically emitted by the flake8-naming plugin (error codes prefixed with 'N').
    """

    # ✅ Best Practice: Register metadata for discovery and filtering
    plugin_name = "number_of_naming_issues"
    plugin_tags = ["naming", "style", "flake8-naming", "N", "convention"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Relies on external flake8-naming plugin being installed
    # 🧠 ML Signal: Naming convention adherence reflects code readability and team practices
    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts the number of naming convention issues reported by Flake8 (e.g., from flake8-naming plugin).

        Args:
            flake8_output (List[str]): List of Flake8 diagnostic output lines.
            file_path (str): Path to the file being analysed (unused in this plugin).

        Returns:
            int: Count of naming convention issues (N-prefixed error codes).
        """
        return sum(1 for line in flake8_output if " N" in line)

    # ✅ Best Practice: High confidence when flake8 output is accessible
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0 if flake8_output else 0.0

    # ✅ Best Practice: Severity scaled by naming violations volume
    def severity_level(self, flake8_output: List[str]) -> str:
        count = self.extract(flake8_output, file_path="")
        if count >= 10:
            return "high"
        elif count >= 3:
            return "medium"
        return "low"
