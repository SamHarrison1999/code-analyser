from typing import List, Dict, Any
from metrics.lizard_metrics.plugins.base import LizardMetricPlugin

class AverageFunctionLengthMetricPlugin(LizardMetricPlugin):
    def name(self) -> str:
        return "average_function_length"

    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        lengths = [m["value"] for m in lizard_metrics if m.get("name") == "average_function_length"]
        return round(sum(lengths) / len(lengths), 2) if lengths else 0.0