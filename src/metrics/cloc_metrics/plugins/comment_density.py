from metrics.cloc_metrics.plugins.base import BaseClocMetricPlugin


class CommentDensityPlugin(BaseClocMetricPlugin):
    def name(self) -> str:
        return "comment_density"

    def extract(self, cloc_data: dict) -> float:
        comment = cloc_data.get("comment", 0)
        blank = cloc_data.get("blank", 0)
        code = cloc_data.get("code", 0)
        total = comment + blank + code
        return round(comment / total, 4) if total > 0 else 0.0
