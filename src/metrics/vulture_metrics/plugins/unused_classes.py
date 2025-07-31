# File: code_analyser/src/metrics/vulture_metrics/plugins/unused_classes.py

from .base import VultureMetricPlugin


class UnusedClassesPlugin(VultureMetricPlugin):
    """
    Counts the number of unused classes detected by Vulture.
    """

    # ✅ Metadata for discovery, tagging, and filtering
    plugin_name = "unused_classes"
    plugin_tags = ["unused", "classes", "vulture", "dead_code"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, vulture_items: list) -> int:
        return sum(1 for item in vulture_items if getattr(item, "typ", "") == "class")

    def confidence_score(self, vulture_items: list) -> float:
        """
        Confidence decreases if very few class candidates are found.
        """
        total_classes = sum(
            1 for item in vulture_items if getattr(item, "typ", "") == "class"
        )
        return 1.0 if total_classes > 0 else 0.0

    def severity_level(self, vulture_items: list) -> str:
        """
        Classify based on number of unused classes.
        """
        count = self.extract(vulture_items)
        if count == 0:
            return "low"
        elif count <= 3:
            return "medium"
        else:
            return "high"
