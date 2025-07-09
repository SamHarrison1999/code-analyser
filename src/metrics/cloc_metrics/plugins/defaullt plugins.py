"""
Defines the default plugin implementations for extracting CLOC metrics.

Each plugin extracts a specific metric from the CLOC JSON output, such as
number of comments, total lines, source lines, and comment density.
"""

from metrics.cloc_metrics.plugins.base import BaseClocMetricPlugin


class CommentCountPlugin(BaseClocMetricPlugin):
    def name(self) -> str:
        return "number_of_comments"

    def extract(self, cloc_data: dict) -> int:
        return cloc_data.get("comment", 0)


class TotalLinesPlugin(BaseClocMetricPlugin):
    def name(self) -> str:
        return "number_of_lines"

    def extract(self, cloc_data: dict) -> int:
        blank = cloc_data.get("blank", 0)
        comment = cloc_data.get("comment", 0)
        code = cloc_data.get("code", 0)
        return blank + comment + code


class SourceLinesPlugin(BaseClocMetricPlugin):
    def name(self) -> str:
        return "number_of_source_lines_of_code"

    def extract(self, cloc_data: dict) -> int:
        return cloc_data.get("code", 0)


class CommentDensityPlugin(BaseClocMetricPlugin):
    def name(self) -> str:
        return "comment_density"

    def extract(self, cloc_data: dict) -> float:
        comment = cloc_data.get("comment", 0)
        blank = cloc_data.get("blank", 0)
        code = cloc_data.get("code", 0)
        total = comment + blank + code
        return round(comment / total, 4) if total > 0 else 0.0
