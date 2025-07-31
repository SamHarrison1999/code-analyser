from typing import List, Dict, Any
from .base import PylintMetricPlugin


class ErrorCountPlugin(PylintMetricPlugin):
    """
    Counts the number of Pylint-reported errors.

    Errors reported by Pylint are typically serious and should be prioritised for fixing.
    """

    plugin_name = "pylint.error"
    plugin_tags = ["error", "critical", "pylint"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> int:
        return sum(1 for issue in pylint_output if issue.get("type") == "error")

    def confidence_score(self, pylint_output: List[Dict[str, Any]]) -> float:
        count = self.extract(pylint_output, file_path="")
        return 1.0 if count > 0 else 0.0

    def severity_level(self, pylint_output: List[Dict[str, Any]]) -> str:
        count = self.extract(pylint_output, file_path="")
        if count >= 5:
            return "high"
        elif count >= 2:
            return "medium"
        return "low"
