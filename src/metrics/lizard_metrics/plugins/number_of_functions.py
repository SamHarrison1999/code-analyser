from typing import List, Dict, Any
from metrics.lizard_metrics.plugins.base import LizardMetricPlugin


class CountFunctionDefinitionsMetricPlugin(LizardMetricPlugin):
    def name(self) -> str:
        return "number_of_functions"

    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> int:
        for m in lizard_metrics:
            if m.get("name") == "number_of_functions":
                return int(m["value"])
        return 0