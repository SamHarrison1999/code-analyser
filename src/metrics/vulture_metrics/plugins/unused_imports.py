# File: code_analyser/src/metrics/vulture_metrics/plugins/unused_imports.py

from .base import VultureMetricPlugin


class UnusedImportsPlugin(VultureMetricPlugin):
    """
    Counts the number of unused import statements detected by Vulture.
    """

    # ✅ Metadata for discovery, tagging, and filtering
    plugin_name = "unused_imports"
    plugin_tags = ["unused", "imports", "vulture", "dead_code"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, vulture_items: list) -> int:
        return sum(1 for item in vulture_items if getattr(item, "typ", "") == "import")

    def confidence_score(self, vulture_items: list) -> float:
        """
        High confidence if at least one unused import is detected.
        """
        count = self.extract(vulture_items)
        return 1.0 if count > 0 else 0.0

    def severity_level(self, vulture_items: list) -> str:
        """
        Classify based on number of unused imports.
        """
        count = self.extract(vulture_items)
        if count == 0:
            return "low"
        elif count <= 3:
            return "medium"
        else:
            return "high"
