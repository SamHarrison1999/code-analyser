# File: code_analyser/src/metrics/vulture_metrics/plugins/unused_functions.py

from .base import VultureMetricPlugin


class UnusedFunctionsPlugin(VultureMetricPlugin):
    """
    Counts the number of unused functions detected by Vulture.
    """

    # ✅ Metadata for discovery, tagging, and filtering
    plugin_name = "unused_functions"
    plugin_tags = ["unused", "functions", "vulture", "dead_code"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, vulture_items: list) -> int:
        return sum(
            1 for item in vulture_items if getattr(item, "typ", "") == "function"
        )

    def confidence_score(self, vulture_items: list) -> float:
        """
        Confidence score is high if unused functions are found, otherwise low.
        """
        count = self.extract(vulture_items)
        return 1.0 if count > 0 else 0.0

    def severity_level(self, vulture_items: list) -> str:
        """
        Classifies the impact based on number of unused functions.
        """
        count = self.extract(vulture_items)
        if count == 0:
            return "low"
        elif count <= 5:
            return "medium"
        else:
            return "high"
