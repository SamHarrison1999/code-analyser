# File: code_analyser/src/metrics/bandit_metrics/plugins/cwe_name_frequency_plugin.py

from typing import Dict, Any
from .base import BanditMetricPlugin


# 🧠 ML Signal: Diverse CWE names help classify project risk patterns
# ⚠️ SAST Risk: More CWE name types → broader unresolved vulnerability types
# ✅ Best Practice: Use plugin metadata, severity, and confidence for overlays
class CWENameFrequencyPlugin(BanditMetricPlugin):
    """
    Builds a frequency map of CWE names (descriptions) found in Bandit results.

    Returns:
        int: Number of unique CWE names.
    """

    plugin_name = "number_of_distinct_cwe_names"
    plugin_tags = ["security", "cwe", "bandit", "description", "nlp"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, data: Dict[str, Any]) -> int:
        names = {
            item.get("cwe", {}).get("name", "").strip()
            for item in data.get("results", [])
            if isinstance(item.get("cwe", {}), dict) and item.get("cwe", {}).get("name")
        }
        return len(names)

    def confidence_score(self, data: Dict[str, Any]) -> float:
        count = self.extract(data)
        return min(1.0, count / 5.0)

    def severity_level(self, data: Dict[str, Any]) -> str:
        count = self.extract(data)
        if count == 0:
            return "low"
        elif count <= 3:
            return "medium"
        else:
            return "high"
