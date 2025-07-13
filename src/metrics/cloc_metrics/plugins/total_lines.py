
from metrics.cloc_metrics.plugins.base import ClocMetricPlugin

class TotalLinesPlugin(ClocMetricPlugin):
    """
    Computes the total number of lines (blank + comment + code)
    as reported by CLOC.

    Returns:
        int: Sum of blank, comment, and code lines.
    """

    def name(self) -> str:
        return "number_of_lines"

    def extract(self, cloc_data: dict) -> int:
        try:
            blank = int(cloc_data.get("blank", 0))
            comment = int(cloc_data.get("comment", 0))
            code = int(cloc_data.get("code", 0))
            return blank + comment + code
        except (TypeError, ValueError):
            return 0
