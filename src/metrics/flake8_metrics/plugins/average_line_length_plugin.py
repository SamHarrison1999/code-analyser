# File: code_analyser/src/metrics/flake8_metrics/plugins/average_line_length_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class AverageLineLengthPlugin(Flake8MetricPlugin):
    """
    Computes the average length of all lines in the source file,
    regardless of Flake8 diagnostics. Useful for measuring style and readability.
    """

    # ✅ Best Practice: Register plugin metadata for discovery
    plugin_name = "average_line_length"
    plugin_tags = ["style", "length", "readability"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: File access may raise exceptions if file is unreadable or missing
    # 🧠 ML Signal: Line length can correlate with readability and adherence to style guides
    def extract(self, flake8_output: List[str], file_path: str) -> float:
        try:
            # Compute line lengths from the source file
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                return 0.0
            total_length = sum(len(line.rstrip("\n")) for line in lines)
            return round(total_length / len(lines), 2)
        except Exception:
            return 0.0

    # ✅ Best Practice: Confidence depends on file readability
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0  # Always computed from actual file content

    # ✅ Best Practice: Severity based on deviation from standard length (e.g. 79 chars)
    def severity_level(self, flake8_output: List[str]) -> str:
        avg = self.extract(flake8_output, file_path="")
        if avg > 100:
            return "high"
        elif avg > 80:
            return "medium"
        return "low"
