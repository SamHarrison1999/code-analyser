from .base import VultureMetricPlugin


class UnusedImportsPlugin(VultureMetricPlugin):
    def name(self) -> str:
        return "unused_imports"

    def extract(self, vulture_items: list) -> int:
        return sum(1 for item in vulture_items if item.typ == "import")
