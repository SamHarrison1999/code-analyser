# File: code_analyser/src/metrics/pydocstyle_metrics/plugins/total_violations_plugin.py

from typing import List
from .base import PydocstyleMetricPlugin


class TotalViolationsMetricPlugin(PydocstyleMetricPlugin):
    """
    Counts the total number of Pydocstyle violations reported for the file.

    Useful as an overall measure of docstring compliance.
    """

    # ✅ Best Practice: Plugin metadata for ML filtering and dashboards
    plugin_name = "number_of_pydocstyle_violations"
    plugin_tags = ["pydocstyle", "docstring", "violations", "total"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, pydocstyle_output: List[str], file_path: str) -> int:
        return len(pydocstyle_output)

    def confidence_score(self, pydocstyle_output: List[str]) -> float:
        # ✅ Confidence increases with number of violations observed
        return min(1.0, len(pydocstyle_output) / 10) if pydocstyle_output else 0.5

    def severity_level(self, pydocstyle_output: List[str]) -> str:
        count = len(pydocstyle_output)
        if count >= 10:
            return "high"
        elif count >= 4:
            return "medium"
        return "low"
