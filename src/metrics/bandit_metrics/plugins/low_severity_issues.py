# File: code_analyser/src/metrics/bandit_metrics/plugins/low_severity_issues.py

from .base import BanditMetricPlugin


# 🧠 ML Signal: Low-severity signals are useful for baseline security metrics and model balance
# ⚠️ SAST Risk: Accumulation of low issues may still indicate systemic risk
# ✅ Best Practice: Tag and expose severity/confidence for visualisation
class LowSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues with LOW severity.

    Returns:
        int: Count of issues marked as LOW severity.
    """

    plugin_name = "number_of_low_security_vulnerabilities"
    plugin_tags = ["security", "bandit", "severity", "low"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, data: dict) -> int:
        return sum(
            1
            for item in data.get("results", [])
            if item.get("issue_severity", "").strip().upper() == "LOW"
        )

    def confidence_score(self, data: dict) -> float:
        count = self.extract(data)
        return min(1.0, 0.2 + (0.1 * count))  # Lower base confidence

    def severity_level(self, data: dict) -> str:
        count = self.extract(data)
        if count == 0:
            return "low"
        elif count <= 2:
            return "low"
        elif count <= 5:
            return "medium"
        else:
            return "high"
