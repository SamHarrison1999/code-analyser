# File: code_analyser/src/metrics/lizard_metrics/plugins/average_function_length.py

from typing import List, Dict, Any
from metrics.lizard_metrics.plugins.base import LizardMetricPlugin


class AverageFunctionLengthMetricPlugin(LizardMetricPlugin):
    """
    Computes the average number of lines per function as reported by Lizard.
    """

    # ✅ Best Practice: Metadata for plugin discovery and tag filtering
    plugin_name = "average_function_length"
    plugin_tags = ["length", "function", "lizard", "average"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Assumes Lizard entries contain valid numeric "value" fields
    # 🧠 ML Signal: Longer average function lengths may correlate with reduced readability
    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        lengths = [
            m["value"]
            for m in lizard_metrics
            if m.get("name") == self.plugin_name
            and isinstance(m.get("value"), (int, float))
        ]
        return round(sum(lengths) / len(lengths), 2) if lengths else 0.0

    # ✅ Best Practice: Confidence is high if valid values were found
    def confidence_score(self, lizard_metrics: List[Dict[str, Any]]) -> float:
        count = sum(1 for m in lizard_metrics if m.get("name") == self.plugin_name)
        return 1.0 if count > 0 else 0.0

    # ✅ Best Practice: Classify severity based on average function length
    def severity_level(self, lizard_metrics: List[Dict[str, Any]]) -> str:
        avg = self.extract(lizard_metrics, file_path="")
        if avg >= 50:
            return "high"
        elif avg >= 25:
            return "medium"
        return "low"
