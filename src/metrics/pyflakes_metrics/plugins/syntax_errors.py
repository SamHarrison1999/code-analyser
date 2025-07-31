# File: code_analyser/src/metrics/pyflakes_metrics/plugins/syntax_errors.py

from typing import List
from metrics.pyflakes_metrics.plugins.base import PyflakesMetricPlugin


class SyntaxErrorsMetricPlugin(PyflakesMetricPlugin):
    """
    Counts the number of syntax errors reported by Pyflakes.

    Syntax errors typically halt static analysis early and indicate serious correctness issues.
    """

    plugin_name = "number_of_syntax_errors"
    plugin_tags = ["error", "syntax", "critical"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, pyflakes_output: List[str], file_path: str) -> int:
        return sum(1 for line in pyflakes_output if "syntax error" in line.lower())

    def confidence_score(self, pyflakes_output: List[str]) -> float:
        count = self.extract(pyflakes_output, file_path="")
        return 1.0 if count > 0 else 0.0

    def severity_level(self, pyflakes_output: List[str]) -> str:
        count = self.extract(pyflakes_output, file_path="")
        return "high" if count > 0 else "low"
