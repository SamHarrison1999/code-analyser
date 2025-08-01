# File: code_analyser/src/metrics/cloc_metrics/plugins/total_lines.py

from metrics.cloc_metrics.plugins.base import ClocMetricPlugin
from typing import Any


class TotalLinesPlugin(ClocMetricPlugin):
    """
    Computes the total number of lines (blank + comment + code)
    as reported by CLOC.

    Returns:
        int: Sum of blank, comment, and code lines.
    """

    # ✅ Best Practice: Plugin metadata for discovery and tag-based filtering
    plugin_name = "number_of_lines"
    plugin_tags = ["lines", "total", "cloc", "summary"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Unsanitised cloc_data may cause type conversion issues
    # 🧠 ML Signal: Total LoC can be a strong input feature for codebase size models
    def extract(self, cloc_data: dict[str, Any]) -> int:
        try:
            blank = int(cloc_data.get("blank", 0))
            comment = int(cloc_data.get("comment", 0))
            code = int(cloc_data.get("code", 0))
            return blank + comment + code
        except (TypeError, ValueError):
            return 0

    # ✅ Best Practice: Trust score based on presence of all required keys
    def confidence_score(self, cloc_data: dict[str, Any]) -> float:
        required = {"blank", "comment", "code"}
        return 1.0 if required.issubset(cloc_data) else 0.6

    # ✅ Best Practice: Severity based on line count
    def severity_level(self, cloc_data: dict[str, Any]) -> str:
        total = self.extract(cloc_data)
        if total == 0:
            return "high"
        elif total < 30:
            return "medium"
        return "low"
