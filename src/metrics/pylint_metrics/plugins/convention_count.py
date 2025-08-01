from typing import List, Dict, Any
from metrics.pylint_metrics.plugins.base import PylintMetricPlugin


class ConventionCountPlugin(PylintMetricPlugin):
    """
    Counts the number of 'convention' issues reported by Pylint.

    Convention messages represent non-critical suggestions to improve code readability
    and maintainability, such as naming standards or formatting preferences.
    """

    plugin_name = "pylint.convention"
    plugin_tags = ["style", "convention", "formatting"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> int:
        return sum(1 for issue in pylint_output if issue.get("type") == "convention")

    def confidence_score(self, pylint_output: List[Dict[str, Any]]) -> float:
        count = self.extract(pylint_output, file_path="")
        return min(1.0, count / 5) if count else 0.0

    def severity_level(self, pylint_output: List[Dict[str, Any]]) -> str:
        count = self.extract(pylint_output, file_path="")
        if count >= 10:
            return "medium"
        return "low"
