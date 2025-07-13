from .base import RadonMetricPlugin


class BlankLinesPlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "blank_lines"

    # ✅ Accept both parsed and file_path to match expected plugin signature
    def extract(self, parsed: dict, file_path: str) -> int:
        return parsed.get("blank_lines", 0)
