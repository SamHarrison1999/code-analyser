# File: code_analyser/src/metrics/lizard_metrics/plugins/average_function_complexity.py

from typing import List, Dict, Any
from metrics.lizard_metrics.plugins.base import LizardMetricPlugin


class AverageCyclomaticComplexityMetricPlugin(LizardMetricPlugin):
    """
    Computes the average cyclomatic complexity across all functions
    reported by Lizard for a given file.
    """

    # ✅ Best Practice: Plugin registration metadata
    plugin_name = "average_function_complexity"
    plugin_tags = ["complexity", "lizard", "cyclomatic", "average"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Assumes all entries with matching "name" have valid numeric "value"
    # 🧠 ML Signal: Average complexity is a strong predictor of maintainability and risk
    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        values = [
            m["value"]
            for m in lizard_metrics
            if m.get("name") == self.plugin_name
            and isinstance(m.get("value"), (int, float))
        ]
        return round(sum(values) / len(values), 2) if values else 0.0

    # ✅ Best Practice: Confidence is high when data exists
    def confidence_score(self, lizard_metrics: List[Dict[str, Any]]) -> float:
        count = sum(1 for m in lizard_metrics if m.get("name") == self.plugin_name)
        return 1.0 if count > 0 else 0.0

    # ✅ Best Practice: Severity based on average complexity thresholds
    def severity_level(self, lizard_metrics: List[Dict[str, Any]]) -> str:
        avg = self.extract(lizard_metrics, file_path="")
        if avg >= 10:
            return "high"
        elif avg >= 5:
            return "medium"
        return "low"
