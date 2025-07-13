from typing import List, Dict, Any
from .base import PylintMetricPlugin


class ConventionCountPlugin(PylintMetricPlugin):
    @staticmethod
    def name() -> str:
        return "pylint.convention"

    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> int:
        return sum(1 for issue in pylint_output if issue.get("type") == "convention")
