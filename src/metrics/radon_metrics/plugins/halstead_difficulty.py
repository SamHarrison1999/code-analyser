# File: code_analyser/src/metrics/radon_metrics/plugins/halstead_difficulty.py

from .base import RadonMetricPlugin


class HalsteadDifficultyPlugin(RadonMetricPlugin):
    """
    Computes the Halstead difficulty metric from Radon output.

    The Halstead difficulty is a measure of how difficult it is to write or understand the code,
    based on the number of unique operators and operands.

    Returns a rounded float to 2 decimal places.
    """

    plugin_name = "halstead_difficulty"
    plugin_tags = ["halstead", "complexity", "difficulty"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, parsed: dict, file_path: str) -> float:
        return round(parsed.get("halstead_difficulty", 0.0), 2)

    def confidence_score(self, parsed: dict) -> float:
        return 1.0 if "halstead_difficulty" in parsed else 0.0

    def severity_level(self, parsed: dict) -> str:
        value = parsed.get("halstead_difficulty", 0.0)
        if value >= 20:
            return "high"
        elif value >= 10:
            return "medium"
        return "low"
