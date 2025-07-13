from typing import List, Dict, Any
from metrics.lizard_metrics.plugins.base import LizardMetricPlugin


class MaintainabilityIndexMetricPlugin(LizardMetricPlugin):
    def name(self) -> str:
        return "maintainability_index"

    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        for m in lizard_metrics:
            if m.get("name") == "maintainability_index":
                return float(m["value"])
        return 0.0
