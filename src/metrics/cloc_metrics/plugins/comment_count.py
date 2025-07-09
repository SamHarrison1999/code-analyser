from metrics.cloc_metrics.plugins.base import BaseClocMetricPlugin

print("✅ Loaded CommentCountPlugin")

class CommentCountPlugin(BaseClocMetricPlugin):
    def name(self) -> str:
        return "number_of_comments"

    def extract(self, cloc_data: dict) -> int:
        return cloc_data.get("comment", 0)
