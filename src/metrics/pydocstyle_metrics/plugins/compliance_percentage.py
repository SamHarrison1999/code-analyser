# File: code_analyser/src/metrics/pydocstyle_metrics/plugins/compliance_percentage.py

from typing import List
from .base import PydocstyleMetricPlugin


class CompliancePercentageMetricPlugin(PydocstyleMetricPlugin):
    """
    Computes the percentage of docstring compliance (i.e., lines without 'Missing docstring').

    A score of 100.0 indicates full compliance.
    """

    # ✅ Best Practice: Plugin metadata for registration and GUI filtering
    plugin_name = "percentage_of_compliance_with_docstring_style"
    plugin_tags = ["docstring", "compliance", "pydocstyle", "quality"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, pydocstyle_output: List[str], file_path: str) -> float:
        total = len(pydocstyle_output)
        missing = sum(1 for line in pydocstyle_output if "Missing docstring" in line)
        if total == 0:
            return 100.0  # ✅ Fully compliant when no violations are reported
        return round(((total - missing) / total) * 100, 2)

    def confidence_score(self, pydocstyle_output: List[str]) -> float:
        # ✅ More lines = higher confidence in the sample
        total = len(pydocstyle_output)
        return min(1.0, total / 20) if total else 0.5

    def severity_level(self, pydocstyle_output: List[str]) -> str:
        value = self.extract(pydocstyle_output, file_path="")
        if value < 50:
            return "high"
        elif value < 80:
            return "medium"
        return "low"
