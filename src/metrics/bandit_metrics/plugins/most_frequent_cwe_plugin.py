from collections import Counter
from typing import Dict, Any, List, Tuple
from .base import BanditMetricPlugin


class MostFrequentCWEPlugin(BanditMetricPlugin):
    """
    Returns the most frequent CWE ID as an integer.

    Returns:
        int: Most common CWE ID, or 0 if none found.
    """
    def name(self) -> str:
        return "most_frequent_cwe"

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
