from collections import Counter
from typing import Dict, Any, List, Tuple
from .base import BanditMetricPlugin

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

        for cid, name in cwe_records:
            if cid == most_common_id:
                try:
                    return {"id": int(cid), "name": name}
                except ValueError:
                    return {"id": 0, "name": name}

        return {"id": 0, "name": "Unknown"}