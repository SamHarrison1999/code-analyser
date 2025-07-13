from collections import Counter
from typing import Dict, Any, List, Tuple
from .base import BanditMetricPlugin

class CWEFrequencyPlugin(BanditMetricPlugin):
    """
    Counts the number of distinct CWE IDs in Bandit results.

    Returns:
        int: Number of unique CWE IDs.
    """
    def name(self) -> str:
        return "number_of_distinct_cwes"

    def extract(self, data: Dict[str, Any]) -> int:
        cwe_ids = {
            str(item.get("cwe", {}).get("id"))
            for item in data.get("results", [])
            if isinstance(item.get("cwe", {}), dict) and item.get("cwe", {}).get("id")
        }
        return len(cwe_ids)