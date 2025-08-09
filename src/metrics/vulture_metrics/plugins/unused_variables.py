# File: code_analyser/src/metrics/vulture_metrics/plugins/unused_variables.py

from .base import VultureMetricPlugin


class UnusedVariablesPlugin(VultureMetricPlugin):
    """
    Counts the number of unused variables detected by Vulture.
    """

    # ✅ Metadata for discovery, tagging, and filtering
    plugin_name = "unused_variables"
    plugin_tags = ["unused", "variables", "vulture", "dead_code"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, vulture_items: list) -> int:
        return sum(1 for item in vulture_items if getattr(item, "typ", "") == "variable")

    def confidence_score(self, vulture_items: list) -> float:
        """
        Returns high confidence if unused variables are found.
        """
        count = self.extract(vulture_items)
        return 1.0 if count > 0 else 0.0

    def severity_level(self, vulture_items: list) -> str:
        """
        Severity increases with the number of unused variables.
        """
        count = self.extract(vulture_items)
        if count == 0:
            return "low"
        elif count <= 4:
            return "medium"
        else:
            return "high"
