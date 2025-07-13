from .base import RadonMetricPlugin


class DocstringLinesPlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "docstring_lines"

    # ✅ Updated to accept both 'parsed' and 'file_path' as required
    def extract(self, parsed: dict, file_path: str) -> int:
        return parsed.get("docstring_lines", 0)
