from metrics.cloc_metrics.plugins.base import BaseClocMetricPlugin

class TotalLinesPlugin(BaseClocMetricPlugin):
    def name(self) -> str:
        return "number_of_lines"

    def extract(self, cloc_data: dict) -> int:
        blank = cloc_data.get("blank", 0)
        comment = cloc_data.get("comment", 0)
        code = cloc_data.get("code", 0)
        return blank + comment + code
