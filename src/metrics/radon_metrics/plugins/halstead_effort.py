from .base import RadonMetricPlugin


class HalsteadEffortPlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "halstead_effort"

    # ✅ Corrected signature to accept parsed and file_path
    def extract(self, parsed: dict, file_path: str) -> float:
        return round(parsed.get("halstead_effort", 0.0), 2)
