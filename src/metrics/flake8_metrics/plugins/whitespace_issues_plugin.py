from .base import Flake8MetricPlugin
from typing import List


class WhitespaceIssuesPlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "number_of_whitespace_issues"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Count the number of whitespace-related issues in Flake8 output.

        Args:
            flake8_output (List[str]): Parsed Flake8 output lines.
            file_path (str): Path to the analysed file (not used directly).

        Returns:
            int: Count of lines reporting whitespace issues (typically E2xx or W2xx codes).
        """
        return sum(
            1 for line in flake8_output
            if any(code in line for code in [": E2", ": W2"])
        )
