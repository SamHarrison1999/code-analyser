from .base import Flake8MetricPlugin
from typing import List


class StylingErrorPlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "number_of_styling_errors"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts the number of Flake8 styling errors related to error codes starting with 'E' or 'F'.

        Args:
            flake8_output (List[str]): List of Flake8 output lines.
            file_path (str): Path to the analysed file (unused in this plugin).

        Returns:
            int: Count of styling errors (E* or F* codes).
        """
        return sum(
            1 for line in flake8_output
            if any(f" {prefix}" in line for prefix in ("E", "F"))
        )
