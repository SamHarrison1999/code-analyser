# File: code_analyser/src/metrics/pylint_metrics/plugins/warning_count.py

from typing import List, Dict, Any
from .base import PylintMetricPlugin


class WarningCountPlugin(PylintMetricPlugin):
    """
    Counts the number of warnings emitted by Pylint.

    These warnings typically indicate possible code issues, maintainability problems,
    or stylistic deviations, but do not prevent execution.
    """

    plugin_name = "pylint.warning"
    plugin_tags = ["warning", "style", "maintainability"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> int:
        return sum(1 for issue in pylint_output if issue.get("type") == "warning")

    def confidence_score(self, pylint_output: List[Dict[str, Any]]) -> float:
        count = self.extract(pylint_output, file_path="")
        return 1.0 if count > 0 else 0.0

    def severity_level(self, pylint_output: List[Dict[str, Any]]) -> str:
        count = self.extract(pylint_output, file_path="")
        if count >= 10:
            return "high"
        elif count >= 5:
            return "medium"
        return "low"
