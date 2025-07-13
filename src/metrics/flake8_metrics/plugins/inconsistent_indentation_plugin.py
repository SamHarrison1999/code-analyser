from .base import Flake8MetricPlugin
from typing import List


class InconsistentIndentationPlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "number_of_inconsistent_indentations"

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
