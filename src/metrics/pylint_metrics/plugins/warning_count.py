from typing import List, Dict, Any
from .base import PylintMetricPlugin


class WarningCountPlugin(PylintMetricPlugin):
    @staticmethod
    def name() -> str:
        return "pylint.warning"

    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> int:
        return sum(1 for issue in pylint_output if issue.get("type") == "warning")
