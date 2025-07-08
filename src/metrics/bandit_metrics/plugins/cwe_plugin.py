# File: src/metrics/bandit_metrics/plugins/cwe_plugin.py

from collections import Counter
from .base import BanditMetricPlugin


class CWEFrequencyPlugin(BanditMetricPlugin):
    """
    Aggregates Bandit issues by CWE ID (if available).

    Returns:
        int: Total number of distinct CWE types found.
    """

    def name(self) -> str:
        return "number_of_distinct_cwes"

    def extract(self, data: dict) -> int:
        cwe_ids = set()
        for item in data.get("results", []):
            cwe = item.get("cwe", {})
            if isinstance(cwe, dict):
                cwe_id = cwe.get("id")
                if cwe_id:
                    cwe_ids.add(cwe_id)
        return len(cwe_ids)

class MostFrequentCWEPlugin(BanditMetricPlugin):
    """
    Returns the numeric CWE ID that occurs most frequently (as int).
    """

    def name(self) -> str:
        return "most_frequent_cwe"

    def extract(self, data: dict) -> int:
        cwes = []
        for item in data.get("results", []):
            cwe = item.get("cwe", {})
            if isinstance(cwe, dict) and "id" in cwe:
                cwes.append(cwe["id"])
        if not cwes:
            return 0
        return int(Counter(cwes).most_common(1)[0][0])
