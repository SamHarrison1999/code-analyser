# File: code_analyser/src/metrics/pydocstyle_metrics/plugins/missing_docstrings.py

from typing import List
from .base import PydocstyleMetricPlugin


class MissingDocstringMetricPlugin(PydocstyleMetricPlugin):
    """
    Counts how many Pydocstyle violations relate to missing docstrings.

    Useful for identifying undocumented classes, functions, or methods.
    """

    # ✅ Best Practice: Plugin metadata for GUI filters and ML export
    plugin_name = "number_of_missing_doc_strings"
    plugin_tags = ["docstring", "missing", "documentation", "pydocstyle"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, pydocstyle_output: List[str], file_path: str) -> int:
        return sum(1 for line in pydocstyle_output if "Missing docstring" in line)

    def confidence_score(self, pydocstyle_output: List[str]) -> float:
        # ✅ More total diagnostics = higher confidence
        return min(1.0, len(pydocstyle_output) / 20) if pydocstyle_output else 0.5

    def severity_level(self, pydocstyle_output: List[str]) -> str:
        count = self.extract(pydocstyle_output, file_path="")
        if count >= 10:
            return "high"
        elif count >= 3:
            return "medium"
        return "low"
