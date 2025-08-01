# File: code_analyser/src/metrics/cloc_metrics/plugins/comment_density.py

from metrics.cloc_metrics.plugins.base import ClocMetricPlugin
from typing import Any


class CommentDensityPlugin(ClocMetricPlugin):
    """
    Computes the ratio of comment lines to total lines (comment + blank + code)
    as a float rounded to 4 decimal places.

    Returns:
        float: Comment density (0.0 if total lines is zero).
    """

    # ✅ Best Practice: Explicit plugin identifier and searchable tags
    plugin_name = "comment_density"
    plugin_tags = ["comment", "density", "ratio", "quality"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Division operation assumes valid numerical input for code, blank, comment
    # 🧠 ML Signal: Comment density can be predictive of documentation quality or maintainability
    def extract(self, cloc_data: dict[str, Any]) -> float:
        try:
            comment = float(cloc_data.get("comment", 0))
            blank = float(cloc_data.get("blank", 0))
            code = float(cloc_data.get("code", 0))
            total = comment + blank + code
            return round(comment / total, 4) if total > 0 else 0.0
        except (TypeError, ValueError):
            return 0.0

    # ✅ Best Practice: Confidence reflects whether all relevant keys are present
    def confidence_score(self, cloc_data: dict[str, Any]) -> float:
        required_keys = {"comment", "blank", "code"}
        present_keys = set(cloc_data)
        return 1.0 if required_keys.issubset(present_keys) else 0.6

    # ✅ Best Practice: Assign severity level based on comment sufficiency
    def severity_level(self, cloc_data: dict[str, Any]) -> str:
        density = self.extract(cloc_data)
        if density >= 0.3:
            return "low"
        elif density >= 0.1:
            return "medium"
        else:
            return "high"
