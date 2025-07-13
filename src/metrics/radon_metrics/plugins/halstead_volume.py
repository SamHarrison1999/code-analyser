from .base import RadonMetricPlugin


class HalsteadVolumePlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "halstead_volume"

    # ✅ Fixed: match expected signature (parsed, file_path)
    def extract(self, parsed: dict, file_path: str) -> float:
        return round(parsed.get("halstead_volume", 0.0), 2)
