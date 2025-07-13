from .base import Flake8MetricPlugin
from typing import List


class TrailingWhitespacesPlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "number_of_trailing_whitespaces"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts the number of lines with trailing whitespace errors in Flake8 output.

        Args:
            flake8_output (List[str]): List of Flake8 diagnostic strings.
            file_path (str): Path to the file being analysed (unused here).

        Returns:
            int: Count of trailing whitespace issues (typically E2xx).
        """
        return sum(1 for line in flake8_output if ": W291" in line or ": E2" in line)
