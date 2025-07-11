from metrics.cloc_metrics.plugins.base import CLOCMetricPlugin

class SourceLinesPlugin(CLOCMetricPlugin):
    """
    Extracts the number of source (code) lines from CLOC output.

    Returns:
        int: Number of lines classified as actual code (SLOC).
    """

    def name(self) -> str:
        return "number_of_source_lines_of_code"

    def extract(self, cloc_data: dict) -> int:
        return int(cloc_data.get("code", 0))
