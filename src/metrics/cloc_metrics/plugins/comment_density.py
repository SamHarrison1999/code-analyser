
from metrics.cloc_metrics.plugins.base import ClocMetricPlugin

class CommentDensityPlugin(ClocMetricPlugin):
    """
    Computes the ratio of comment lines to total lines (comment + blank + code)
    as a float rounded to 4 decimal places.

    Returns:
        float: Comment density (0.0 if total lines is zero).
    """

    def name(self) -> str:
        return "comment_density"

    def extract(self, cloc_data: dict) -> float:
        try:
            comment = float(cloc_data.get("comment", 0))
            blank = float(cloc_data.get("blank", 0))
            code = float(cloc_data.get("code", 0))
            total = comment + blank + code
            return round(comment / total, 4) if total > 0 else 0.0
        except (TypeError, ValueError):
            return 0.0
