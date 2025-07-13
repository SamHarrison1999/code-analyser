from typing import List
from .base import PydocstyleMetricPlugin


class CompliancePercentageMetricPlugin(PydocstyleMetricPlugin):
    @classmethod
    def name(cls) -> str:
        return "percentage_of_compliance_with_docstring_style"

    def extract(self, pydocstyle_output: List[str], file_path: str) -> float:
        total = len(pydocstyle_output)
        missing = sum(1 for line in pydocstyle_output if "Missing docstring" in line)
        if total == 0:
            return 100.0
        return round(((total - missing) / total) * 100, 2)
