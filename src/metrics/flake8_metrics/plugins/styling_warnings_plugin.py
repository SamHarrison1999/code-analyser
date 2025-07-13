from .base import Flake8MetricPlugin
from typing import List


class StylingWarningPlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "number_of_styling_warnings"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts the number of Flake8 warnings related to styling (e.g., 'W' codes).

        Args:
            flake8_output (List[str]): List of Flake8 diagnostic lines.
            file_path (str): Path to the source file (unused here).

        Returns:
            int: Count of style-related warnings flagged by Flake8.
        """
        return sum(1 for line in flake8_output if ": W" in line)
