from .base import RadonMetricPlugin


class HalsteadDifficultyPlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "halstead_difficulty"

    # ✅ Updated to accept both 'parsed' and 'file_path' to match plugin interface
    def extract(self, parsed: dict, file_path: str) -> float:
        return round(parsed.get("halstead_difficulty", 0.0), 2)
