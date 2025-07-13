from .base import Flake8MetricPlugin
from typing import List


class TotalIssuePlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "number_of_styling_issues"

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
