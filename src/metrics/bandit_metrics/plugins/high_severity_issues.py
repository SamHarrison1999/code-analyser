# File: code_analyser/src/metrics/bandit_metrics/plugins/high_severity_issues.py

from .base import BanditMetricPlugin


# 🧠 ML Signal: High-severity Bandit issues strongly correlate with critical code risk
# ⚠️ SAST Risk: These issues are high-priority for manual auditing and patching
# ✅ Best Practice: Register with tags and confidence heuristics
class HighSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues with HIGH severity.

    Returns:
        int: Count of issues marked as HIGH severity.
    """

    plugin_name = "number_of_high_security_vulnerabilities"
    plugin_tags = ["security", "bandit", "severity", "high"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, data: dict) -> int:
        return sum(
            1
            for item in data.get("results", [])
            if item.get("issue_severity", "").strip().upper() == "HIGH"
        )

    def confidence_score(self, data: dict) -> float:
        count = self.extract(data)
        return min(
            1.0, 0.5 + (0.1 * count)
        )  # Boosted confidence for critical vulnerabilities

    def severity_level(self, data: dict) -> str:
        count = self.extract(data)
        if count == 0:
            return "low"
        elif count == 1:
            return "medium"
        else:
            return "high"
