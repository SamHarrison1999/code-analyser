# File: code_analyser/src/metrics/lizard_metrics/plugins/average_parameter_count.py

from typing import List, Dict, Any
from metrics.lizard_metrics.plugins.base import LizardMetricPlugin


class AverageParameterCountMetricPlugin(LizardMetricPlugin):
    """
    Calculates the average number of parameters per function as reported by Lizard.
    """

    # ✅ Best Practice: Metadata for plugin discovery, filtering, and display
    plugin_name = "average_parameter_count"
    plugin_tags = ["parameters", "function", "lizard", "average"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Ensure only numeric values are processed
    # 🧠 ML Signal: High parameter counts may indicate poor cohesion or excessive coupling
    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        values = [
            m["value"]
            for m in lizard_metrics
            if m.get("name") == self.plugin_name
            and isinstance(m.get("value"), (int, float))
        ]
        return round(sum(values) / len(values), 2) if values else 0.0

    # ✅ Best Practice: Confidence score reflects whether metric data was found
    def confidence_score(self, lizard_metrics: List[Dict[str, Any]]) -> float:
        count = sum(1 for m in lizard_metrics if m.get("name") == self.plugin_name)
        return 1.0 if count > 0 else 0.0

    # ✅ Best Practice: Severity reflects potential maintainability risks
    def severity_level(self, lizard_metrics: List[Dict[str, Any]]) -> str:
        avg = self.extract(lizard_metrics, file_path="")
        if avg >= 6:
            return "high"
        elif avg >= 3:
            return "medium"
        return "low"
