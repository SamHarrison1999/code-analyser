"""
Default Lizard metric plugins.

Each plugin extracts one metric from the pre-parsed Lizard output.
"""

from typing import List, Dict, Any
from metrics.lizard_metrics.plugins.base import LizardPlugin


class AverageFunctionLengthPlugin(LizardPlugin):
    @classmethod
    def name(cls) -> str:
        return "average_function_length"

    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        lengths = [m["value"] for m in lizard_metrics if m.get("name") == "average_function_length"]
        return round(sum(lengths) / len(lengths), 2) if lengths else 0.0


class AverageCyclomaticComplexityPlugin(LizardPlugin):
    @classmethod
    def name(cls) -> str:
        return "average_function_complexity"

    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        values = [m["value"] for m in lizard_metrics if m.get("name") == "average_function_complexity"]
        return round(sum(values) / len(values), 2) if values else 0.0


class MaintainabilityIndexPlugin(LizardPlugin):
    @classmethod
    def name(cls) -> str:
        return "maintainability_index"

    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        for m in lizard_metrics:
            if m.get("name") == "maintainability_index":
                return float(m["value"])
        return 0.0


class AverageParameterCountPlugin(LizardPlugin):
    @classmethod
    def name(cls) -> str:
        return "average_parameter_count"

    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        values = [m["value"] for m in lizard_metrics if m.get("name") == "average_parameter_count"]
        return round(sum(values) / len(values), 2) if values else 0.0


class AverageTokenCountPlugin(LizardPlugin):
    @classmethod
    def name(cls) -> str:
        return "average_token_count"

    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> float:
        values = [m["value"] for m in lizard_metrics if m.get("name") == "average_token_count"]
        return round(sum(values) / len(values), 2) if values else 0.0


def load_plugins() -> List[LizardPlugin]:
    """
    Loads all default Lizard metric plugins.

    Returns:
        List[LizardPlugin]: List of plugin instances.
    """
    return [
        AverageCyclomaticComplexityPlugin(),
        AverageTokenCountPlugin(),
        AverageParameterCountPlugin(),
        AverageFunctionLengthPlugin(),
        CountFunctionDefinitionsPlugin(),
        MaintainabilityIndexPlugin(),
    ]


class CountFunctionDefinitionsPlugin(LizardPlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_functions"

    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> int:
        for m in lizard_metrics:
            if m.get("name") == "number_of_functions":
                return int(m["value"])
        return 0


# ✅ Exported list of plugin instances for unified access
plugins: List[LizardPlugin] = load_plugins()
