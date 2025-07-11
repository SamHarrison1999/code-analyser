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


class MostFrequentCWEWithNamePlugin(BanditMetricPlugin):
    """
    Returns the most frequent CWE ID along with its name/description.

    Returns:
        dict: {'id': <int>, 'name': <str>} or {'id': 0, 'name': 'Unknown'}
    """

    def name(self) -> str:
        return "most_frequent_cwe_with_name"

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

        # Find first matching name
        for cid, name in cwe_records:
            if cid == most_common_id:
                try:
                    return {"id": int(cid), "name": name}
                except ValueError:
                    return {"id": 0, "name": name}

        return {"id": 0, "name": "Unknown"}
