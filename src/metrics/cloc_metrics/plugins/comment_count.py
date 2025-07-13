
from metrics.cloc_metrics.plugins.base import ClocMetricPlugin

class CommentCountPlugin(ClocMetricPlugin):
    """
    Extracts the total number of comment lines from CLOC output.

    Returns:
        int: Number of comment lines across all languages.
    """

    def name(self) -> str:
        return "number_of_comments"

    def extract(self, cloc_data: dict) -> int:
        try:
            return int(cloc_data.get("comment", 0))
        except (TypeError, ValueError):
            return 0
