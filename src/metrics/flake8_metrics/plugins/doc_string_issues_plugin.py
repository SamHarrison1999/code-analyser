from .base import Flake8MetricPlugin
from typing import List


class DocstringIssuePlugin(Flake8MetricPlugin):
    def name(self) -> str:
        return "number_of_doc_string_issues"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts how many Flake8 diagnostics are related to docstring violations.

        Args:
            flake8_output (List[str]): The lines from Flake8 output.
            file_path (str): Path to the file being analysed.

        Returns:
            int: Number of docstring-related issues (codes starting with 'D').
        """
        return sum(1 for line in flake8_output if ": D" in line or " D" in line)
