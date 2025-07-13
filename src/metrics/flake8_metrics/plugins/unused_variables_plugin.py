from .base import Flake8MetricPlugin
from typing import List


class UnusedVariablePlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "number_of_unused_variables"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Count the number of unused variable warnings (F841) reported by Flake8.

        Args:
            flake8_output (List[str]): Parsed Flake8 output lines.
            file_path (str): Path to the analysed file (not used in logic).

        Returns:
            int: Count of F841 violations for unused variables.
        """
        return sum(1 for line in flake8_output if ": F841 " in line)
