# File: code_analyser/src/metrics/bandit_metrics/plugins/most_frequent_cwe_with_name_plugin.py

from collections import Counter
from typing import Dict, Any, List, Tuple
from .base import BanditMetricPlugin


# 🧠 ML Signal: Most frequent CWE (with name) helps contextualise recurring patterns
# ⚠️ SAST Risk: CWE name repetition may reveal design patterns linked to specific vulnerabilities
# ✅ Best Practice: Plugin supports severity and confidence overlays
class MostFrequentCWEWithNamePlugin(BanditMetricPlugin):
    """
    Returns the most frequent CWE ID along with its name/description.

    Returns:
        dict: {'id': <int>, 'name': <str>} or {'id': 0, 'name': 'Unknown'}
    """

    plugin_name = "most_frequent_cwe_with_name"
    plugin_tags = ["security", "bandit", "cwe", "name", "nlp", "descriptive"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cwe_records: List[Tuple[str, str]] = [
            (str(cwe.get("id")), cwe.get("name", ""))
            for item in data.get("results", [])
            if (cwe := item.get("cwe")) and isinstance(cwe, dict) and cwe.get("id")
        ]

        if not cwe_records:
            return {"id": 0, "name": "Unknown"}

        ids = [record[0] for record in cwe_records]
        most_common_id = Counter(ids).most_common(1)[0][0]

        for cid, name in cwe_records:
            if cid == most_common_id:
                try:
                    return {"id": int(cid), "name": name or "Unknown"}
                except ValueError:
                    return {"id": 0, "name": name or "Unknown"}

        return {"id": 0, "name": "Unknown"}

    def confidence_score(self, data: Dict[str, Any]) -> float:
        ids = [
            str(item.get("cwe", {}).get("id"))
            for item in data.get("results", [])
            if isinstance(item.get("cwe", {}), dict) and item.get("cwe", {}).get("id")
        ]
        if not ids:
            return 0.0
        most_common = Counter(ids).most_common(1)[0][1]
        total = len(ids)
        return round(min(1.0, most_common / max(1, total)), 2)

    def severity_level(self, data: Dict[str, Any]) -> str:
        try:
            result = self.extract(data)
            cwe_id = int(result.get("id", 0))
        except Exception:
            return "low"

        if cwe_id >= 1000:
            return "low"
        elif cwe_id >= 500:
            return "medium"
        elif cwe_id > 0:
            return "high"
        return "low"
