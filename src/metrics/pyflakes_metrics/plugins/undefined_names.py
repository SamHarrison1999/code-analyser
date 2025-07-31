# File: code_analyser/src/metrics/pyflakes_metrics/plugins/undefined_names.py

from typing import List
from metrics.pyflakes_metrics.plugins.base import PyflakesMetricPlugin


class UndefinedNamesMetricPlugin(PyflakesMetricPlugin):
    """
    Counts the number of 'undefined name' diagnostics reported by Pyflakes.

    This metric highlights unresolved symbols which can lead to runtime NameErrors.
    """

    plugin_name = "number_of_undefined_names"
    plugin_tags = ["error", "undefined", "name", "symbol"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, pyflakes_output: List[str], file_path: str) -> int:
        return sum(1 for line in pyflakes_output if "undefined name" in line.lower())

    def confidence_score(self, pyflakes_output: List[str]) -> float:
        count = self.extract(pyflakes_output, file_path="")
        return min(1.0, count / 5.0) if count > 0 else 0.0

    def severity_level(self, pyflakes_output: List[str]) -> str:
        count = self.extract(pyflakes_output, file_path="")
        if count >= 5:
            return "high"
        elif count >= 2:
            return "medium"
        return "low"
