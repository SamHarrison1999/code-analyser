# File: code_analyser/src/metrics/bandit_metrics/plugins/medium_severity_issues.py

from .base import BanditMetricPlugin


# 🧠 ML Signal: Medium-severity issues are informative for risk segmentation
# ⚠️ SAST Risk: Often ignored despite comprising majority of exploitable flaws
# ✅ Best Practice: Scaled scoring and metadata tags for GUI/ML export
class MediumSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues with MEDIUM severity.

    Returns:
        int: Count of issues marked as MEDIUM severity.
    """

    plugin_name = "number_of_medium_security_vulnerabilities"
    plugin_tags = ["security", "bandit", "severity", "medium"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, data: dict) -> int:
        return sum(
            1
            for item in data.get("results", [])
            if item.get("issue_severity", "").strip().upper() == "MEDIUM"
        )

    def confidence_score(self, data: dict) -> float:
        count = self.extract(data)
        return min(1.0, 0.3 + (0.1 * count))

    def severity_level(self, data: dict) -> str:
        count = self.extract(data)
        if count == 0:
            return "low"
        elif count <= 2:
            return "medium"
        else:
            return "high"
