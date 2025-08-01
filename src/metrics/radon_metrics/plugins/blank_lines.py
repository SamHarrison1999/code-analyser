# File: code_analyser/src/metrics/radon_metrics/plugins/blank_lines.py

from .base import RadonMetricPlugin


class BlankLinesPlugin(RadonMetricPlugin):
    """
    Computes the number of blank lines in the analysed source file
    as reported by Radon.

    Useful as a formatting/style signal or for line density normalisation.
    """

    plugin_name = "blank_lines"
    plugin_tags = ["formatting", "spacing", "layout"]

    def name(self) -> str:
        return self.plugin_name

    # ✅ Accept both parsed and file_path to match expected plugin signature
    def extract(self, parsed: dict, file_path: str) -> int:
        return parsed.get("blank_lines", 0)

    def confidence_score(self, parsed: dict) -> float:
        return 1.0 if "blank_lines" in parsed else 0.0

    def severity_level(self, parsed: dict) -> str:
        count = parsed.get("blank_lines", 0)
        if count > 100:
            return "medium"
        return "low"
