from collections import Counter
from typing import Dict, Any, List, Tuple
from .base import BanditMetricPlugin

class CWENameFrequencyPlugin(BanditMetricPlugin):
    """
    Builds a frequency map of CWE names (descriptions) found in Bandit results.

    Returns:
        int: Number of unique CWE names.
    """
    def name(self) -> str:
        return "number_of_distinct_cwe_names"

    def extract(self, data: Dict[str, Any]) -> int:
        names = {
            item.get("cwe", {}).get("name", "").strip()
            for item in data.get("results", [])
            if isinstance(item.get("cwe", {}), dict) and item.get("cwe", {}).get("name")
        }
        return len(names)