# File: code_analyser/src/metrics/lizard_metrics/plugins/maintainability_index.py

from typing import List, Dict, Any
from metrics.lizard_metrics.plugins.base import LizardMetricPlugin


class MaintainabilityIndexMetricPlugin(LizardMetricPlugin):
    """
    Extracts the Maintainability Index (MI) for the analysed file as reported by Lizard.
    """

    # ✅ Best Practice: Metadata for registry-based loading and export filtering
    plugin_name = "maintainability_index"
    plugin_tags = ["maintainability", "index", "lizard", "score"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Assumes Lizard includes 'maintainability_index' with a valid numeric value
    # 🧠 ML Signal: MI is a strong predictor of code quality and maintainability
    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        for m in lizard_metrics:
            if m.get("name") == self.plugin_name and isinstance(m.get("value"), (int, float)):
                return float(m["value"])
        return 0.0

    # ✅ Best Practice: Confidence depends on whether MI is reported
    def confidence_score(self, lizard_metrics: List[Dict[str, Any]]) -> float:
        for m in lizard_metrics:
            if m.get("name") == self.plugin_name:
                return 1.0
        return 0.0

    # ✅ Best Practice: Severity scales inversely with maintainability score
    def severity_level(self, lizard_metrics: List[Dict[str, Any]]) -> str:
        mi = self.extract(lizard_metrics, file_path="")
        if mi < 50:
            return "high"
        elif mi < 75:
            return "medium"
        return "low"
