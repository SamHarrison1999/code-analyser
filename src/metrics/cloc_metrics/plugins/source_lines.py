
from metrics.cloc_metrics.plugins.base import ClocMetricPlugin

class SourceLinesPlugin(ClocMetricPlugin):
    """
    Extracts the number of source (code) lines from CLOC output.

    Returns:
        int: Number of lines classified as actual code (SLOC).
    """

    def name(self) -> str:
        return "number_of_source_lines_of_code"

    def extract(self, cloc_data: dict) -> int:
        try:
            return int(cloc_data.get("code", 0))
        except (TypeError, ValueError):
            return 0
