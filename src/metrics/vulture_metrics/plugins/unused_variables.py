from .base import VultureMetricPlugin


class UnusedVariablesPlugin(VultureMetricPlugin):
    def name(self) -> str:
        return "unused_variables"

    def extract(self, vulture_items: list) -> int:
        return sum(1 for item in vulture_items if item.typ == "variable")
