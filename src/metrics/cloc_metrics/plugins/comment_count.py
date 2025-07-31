# File: code_analyser/src/metrics/cloc_metrics/plugins/comment_count.py

from metrics.cloc_metrics.plugins.base import ClocMetricPlugin
from typing import Any


class CommentCountPlugin(ClocMetricPlugin):
    """
    Extracts the total number of comment lines from CLOC output.

    Returns:
        int: Number of comment lines across all languages.
    """

    # ✅ Best Practice: Unique plugin identifier and semantic tags
    plugin_name = "number_of_comments"
    plugin_tags = ["comment", "lines", "summary"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Trust boundary - 'cloc_data' might be malformed or incomplete
    # 🧠 ML Signal: Number of comment lines may correlate with code documentation quality
    def extract(self, cloc_data: dict[str, Any]) -> int:
        try:
            return int(cloc_data.get("comment", 0))
        except (TypeError, ValueError):
            return 0

    # ✅ Best Practice: Provide confidence level for AI integration
    def confidence_score(self, cloc_data: dict[str, Any]) -> float:
        return 1.0 if "comment" in cloc_data else 0.5

    # ✅ Best Practice: Estimate severity as low since lack of comments is not critical
    def severity_level(self, cloc_data: dict[str, Any]) -> str:
        comment_lines = cloc_data.get("comment", 0)
        return "low" if comment_lines > 0 else "medium"
