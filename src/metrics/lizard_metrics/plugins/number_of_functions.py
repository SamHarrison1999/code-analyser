# File: code_analyser/src/metrics/lizard_metrics/plugins/number_of_functions.py

from typing import List, Dict, Any
from metrics.lizard_metrics.plugins.base import LizardMetricPlugin


class CountFunctionDefinitionsMetricPlugin(LizardMetricPlugin):
    """
    Extracts the total number of function definitions in a file as reported by Lizard.
    """

    # ✅ Best Practice: Plugin metadata for dynamic registry and tagging
    plugin_name = "number_of_functions"
    plugin_tags = ["function", "count", "lizard", "structure"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Assumes well-formed numeric 'value' for 'number_of_functions'
    # 🧠 ML Signal: Function count reflects structural complexity and modularity
    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> int:
        for m in lizard_metrics:
            if m.get("name") == self.plugin_name and isinstance(
                m.get("value"), (int, float)
            ):
                return int(m["value"])
        return 0

    # ✅ Best Practice: Confidence is binary based on presence of the metric
    def confidence_score(self, lizard_metrics: List[Dict[str, Any]]) -> float:
        for m in lizard_metrics:
            if m.get("name") == self.plugin_name:
                return 1.0
        return 0.0

    # ✅ Best Practice: Severity may reflect over-decomposition or excessive granularity
    def severity_level(self, lizard_metrics: List[Dict[str, Any]]) -> str:
        count = self.extract(lizard_metrics, file_path="")
        if count >= 50:
            return "high"
        elif count >= 20:
            return "medium"
        return "low"
