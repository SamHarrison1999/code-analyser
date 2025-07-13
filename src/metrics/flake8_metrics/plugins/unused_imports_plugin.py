from .base import Flake8MetricPlugin
from typing import List


class UnusedImportPlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "number_of_unused_imports"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Count the number of unused import warnings (F401) reported by Flake8.

        Args:
            flake8_output (List[str]): Parsed Flake8 output lines.
            file_path (str): Path to the analysed file (not used here).

        Returns:
            int: Number of F401 unused import violations.
        """
        return sum(1 for line in flake8_output if ": F401 " in line)
