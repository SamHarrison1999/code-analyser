# File: code_analyser/src/metrics/lizard_metrics/plugins/average_token_count.py

from typing import List, Dict, Any
from metrics.lizard_metrics.plugins.base import LizardMetricPlugin


class AverageTokenCountMetricPlugin(LizardMetricPlugin):
    """
    Computes the average number of tokens per function based on Lizard analysis.
    """

    # ✅ Best Practice: Plugin metadata for discovery and filterable categorisation
    plugin_name = "average_token_count"
    plugin_tags = ["tokens", "function", "lizard", "average"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Ensure that only numeric values are aggregated
    # 🧠 ML Signal: Token count approximates function length and lexical complexity
    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        values = [
            m["value"]
            for m in lizard_metrics
            if m.get("name") == self.plugin_name and isinstance(m.get("value"), (int, float))
        ]
        return round(sum(values) / len(values), 2) if values else 0.0

    # ✅ Best Practice: Confidence score reflects valid data presence
    def confidence_score(self, lizard_metrics: List[Dict[str, Any]]) -> float:
        count = sum(1 for m in lizard_metrics if m.get("name") == self.plugin_name)
        return 1.0 if count > 0 else 0.0

    # ✅ Best Practice: Severity scaled by lexical complexity per function
    def severity_level(self, lizard_metrics: List[Dict[str, Any]]) -> str:
        avg = self.extract(lizard_metrics, file_path="")
        if avg >= 150:
            return "high"
        elif avg >= 75:
            return "medium"
        return "low"
