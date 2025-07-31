# File: code_analyser/src/metrics/bandit_metrics/plugins/most_frequent_cwe_plugin.py

from collections import Counter
from typing import Dict, Any
from .base import BanditMetricPlugin


# 🧠 ML Signal: Most frequent CWE ID is useful for category-based code risk analysis
# ⚠️ SAST Risk: Repeated CWE types may indicate lack of remediation or poor training
# ✅ Best Practice: Expose frequency, confidence and severity metadata
class MostFrequentCWEPlugin(BanditMetricPlugin):
    """
    Returns the most frequent CWE ID as an integer.

    Returns:
        int: Most common CWE ID, or 0 if none found.
    """

    plugin_name = "most_frequent_cwe"
    plugin_tags = ["security", "bandit", "cwe", "frequency", "categorical"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, data: Dict[str, Any]) -> int:
        cwe_ids = [
            str(item.get("cwe", {}).get("id"))
            for item in data.get("results", [])
            if isinstance(item.get("cwe", {}), dict) and item.get("cwe", {}).get("id")
        ]
        if not cwe_ids:
            return 0
        try:
            return int(Counter(cwe_ids).most_common(1)[0][0])
        except ValueError:
            return 0

    def confidence_score(self, data: Dict[str, Any]) -> float:
        cwe_ids = [
            str(item.get("cwe", {}).get("id"))
            for item in data.get("results", [])
            if isinstance(item.get("cwe", {}), dict) and item.get("cwe", {}).get("id")
        ]
        if not cwe_ids:
            return 0.0
        most_common = Counter(cwe_ids).most_common(1)[0][1]
        total = len(cwe_ids)
        return round(min(1.0, most_common / max(1, total)), 2)

    def severity_level(self, data: Dict[str, Any]) -> str:
        value = self.extract(data)
        if value == 0:
            return "low"
        # Heuristic: Map known CWE ID ranges
        try:
            if value >= 1000:
                return "low"
            elif value >= 500:
                return "medium"
            else:
                return "high"
        except Exception:
            return "medium"
