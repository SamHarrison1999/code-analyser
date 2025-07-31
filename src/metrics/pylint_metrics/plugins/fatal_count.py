# File: code_analyser/src/metrics/pylint_metrics/plugins/fatal_count.py

from typing import List, Dict, Any
from .base import PylintMetricPlugin


class FatalCountPlugin(PylintMetricPlugin):
    """
    Counts the number of fatal errors reported by Pylint.

    Fatal errors represent serious failures such as syntax issues or internal errors that
    prevent Pylint from continuing the analysis.
    """

    plugin_name = "pylint.fatal"
    plugin_tags = ["error", "fatal", "critical"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> int:
        return sum(1 for issue in pylint_output if issue.get("type") == "fatal")

    def confidence_score(self, pylint_output: List[Dict[str, Any]]) -> float:
        count = self.extract(pylint_output, file_path="")
        return 1.0 if count > 0 else 0.0

    def severity_level(self, pylint_output: List[Dict[str, Any]]) -> str:
        count = self.extract(pylint_output, file_path="")
        return "high" if count > 0 else "low"
