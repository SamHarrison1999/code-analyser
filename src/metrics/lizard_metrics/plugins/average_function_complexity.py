from typing import List, Dict, Any
from metrics.lizard_metrics.plugins.base import LizardMetricPlugin

class AverageCyclomaticComplexityMetricPlugin(LizardMetricPlugin):
    def name(self) -> str:
        return "average_function_complexity"

    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        values = [m["value"] for m in lizard_metrics if m.get("name") == "average_function_complexity"]
        return round(sum(values) / len(values), 2) if values else 0.0