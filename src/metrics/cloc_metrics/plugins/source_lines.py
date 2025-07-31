# File: code_analyser/src/metrics/cloc_metrics/plugins/source_lines.py

from metrics.cloc_metrics.plugins.base import ClocMetricPlugin
from typing import Any


class SourceLinesPlugin(ClocMetricPlugin):
    """
    Extracts the number of source (code) lines from CLOC output.

    Returns:
        int: Number of lines classified as actual code (SLOC).
    """

    # ✅ Best Practice: Metadata for plugin discovery and filtering
    plugin_name = "number_of_source_lines_of_code"
    plugin_tags = ["code", "sloc", "size", "lines"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Input from cloc may be malformed; casting must be guarded
    # 🧠 ML Signal: SLOC is a strong signal of file size and potential complexity
    def extract(self, cloc_data: dict[str, Any]) -> int:
        try:
            return int(cloc_data.get("code", 0))
        except (TypeError, ValueError):
            return 0

    # ✅ Best Practice: Confidence based on presence of 'code' key
    def confidence_score(self, cloc_data: dict[str, Any]) -> float:
        return 1.0 if "code" in cloc_data else 0.5

    # ✅ Best Practice: Severity reflects unusually large or zero code files
    def severity_level(self, cloc_data: dict[str, Any]) -> str:
        sloc = cloc_data.get("code", 0)
        if sloc == 0:
            return "high"
        elif sloc < 20:
            return "medium"
        else:
            return "low"
