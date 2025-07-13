from .base import RadonMetricPlugin


class LogicalLinesPlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "logical_lines"

    # ✅ Fixed: match expected extract signature (parsed, file_path)
    def extract(self, parsed: dict, file_path: str) -> int:
        return parsed.get("logical_lines", 0)
