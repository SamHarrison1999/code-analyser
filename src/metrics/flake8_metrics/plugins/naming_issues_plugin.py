from .base import Flake8MetricPlugin
from typing import List


class NamingIssuePlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "number_of_naming_issues"

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
