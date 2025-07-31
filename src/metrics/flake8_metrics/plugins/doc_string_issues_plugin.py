# File: code_analyser/src/metrics/flake8_metrics/plugins/doc_string_issues_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class DocstringIssuePlugin(Flake8MetricPlugin):
    """
    Counts how many Flake8 diagnostics are related to docstring violations.
    This is usually based on codes starting with 'D' (pydocstyle or flake8-docstrings).
    """

    # ✅ Best Practice: Plugin metadata for discovery and filtering
    plugin_name = "number_of_doc_string_issues"
    plugin_tags = ["docstring", "style", "D", "pydocstyle"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Relies on external Flake8 plugin presence (e.g. flake8-docstrings)
    # 🧠 ML Signal: Docstring violations may correlate with poor documentation practices
    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts how many Flake8 diagnostics are related to docstring violations.

        Args:
            flake8_output (List[str]): The lines from Flake8 output.
            file_path (str): Path to the file being analysed.

        Returns:
            int: Number of docstring-related issues (codes starting with 'D').
        """
        return sum(1 for line in flake8_output if ": D" in line or " D" in line)

    # ✅ Best Practice: High confidence when parsing Flake8 diagnostics directly
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0

    # ✅ Best Practice: Assign severity based on volume of missing or invalid docstrings
    def severity_level(self, flake8_output: List[str]) -> str:
        count = self.extract(flake8_output, file_path="")
        if count >= 10:
            return "high"
        elif count >= 3:
            return "medium"
        return "low"
