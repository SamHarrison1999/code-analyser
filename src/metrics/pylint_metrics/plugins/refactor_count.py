# File: code_analyser/src/metrics/pylint_metrics/plugins/refactor_count.py

from typing import List, Dict, Any
from .base import PylintMetricPlugin


class RefactorCountPlugin(PylintMetricPlugin):
    """
    Counts the number of 'refactor' suggestions reported by Pylint.

    These suggestions usually indicate opportunities to improve code structure,
    such as reducing complexity or eliminating duplication.
    """

    plugin_name = "pylint.refactor"
    plugin_tags = ["refactor", "quality", "design"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> int:
        return sum(1 for issue in pylint_output if issue.get("type") == "refactor")

    def confidence_score(self, pylint_output: List[Dict[str, Any]]) -> float:
        count = self.extract(pylint_output, file_path="")
        return min(1.0, count / 10) if count else 0.0

    def severity_level(self, pylint_output: List[Dict[str, Any]]) -> str:
        count = self.extract(pylint_output, file_path="")
        if count >= 5:
            return "high"
        elif count >= 2:
            return "medium"
        return "low"
