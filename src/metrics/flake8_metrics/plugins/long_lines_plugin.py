from .base import Flake8MetricPlugin
from typing import List


class LongLinePlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "number_of_long_lines"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts the number of long line violations (E501) in Flake8 output.

        Args:
            flake8_output (List[str]): List of Flake8 diagnostic output lines.
            file_path (str): Path to the file being analysed (not used in this plugin).

        Returns:
            int: Count of E501 long line violations.
        """
        return sum(1 for line in flake8_output if " E501" in line)
