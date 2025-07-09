from metrics.cloc_metrics.plugins.base import BaseClocMetricPlugin

class SourceLinesPlugin(BaseClocMetricPlugin):
    def name(self) -> str:
        return "number_of_source_lines_of_code"

    def extract(self, cloc_data: dict) -> int:
        return cloc_data.get("code", 0)
