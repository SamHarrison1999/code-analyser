# File: code_analyser/src/metrics/bandit_metrics/plugins/cwe_frequency_plugin.py

from typing import Dict, Any
from .base import BanditMetricPlugin


# 🧠 ML Signal: CWE spread can be used to characterise security risk diversity
# ⚠️ SAST Risk: A high number of distinct CWE IDs implies broader code vulnerability surface
# ✅ Best Practice: Include plugin_name, plugin_tags, and severity scoring
class CWEFrequencyPlugin(BanditMetricPlugin):
    """
    Counts the number of distinct CWE IDs in Bandit results.

    Returns:
        int: Number of unique CWE IDs.
    """

    # ✅ Required metadata for plugin registry
    plugin_name = "number_of_distinct_cwes"
    plugin_tags = ["security", "cwe", "bandit", "diversity"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, data: Dict[str, Any]) -> int:
        cwe_ids = {
            str(item.get("cwe", {}).get("id"))
            for item in data.get("results", [])
            if isinstance(item.get("cwe", {}), dict) and item.get("cwe", {}).get("id")
        }
        return len(cwe_ids)

    def confidence_score(self, data: Dict[str, Any]) -> float:
        count = self.extract(data)
        return min(1.0, count / 5.0)  # Full confidence if 5+ distinct CWEs

    def severity_level(self, data: Dict[str, Any]) -> str:
        count = self.extract(data)
        if count == 0:
            return "low"
        elif count <= 3:
            return "medium"
        else:
            return "high"
