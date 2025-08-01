# File: code_analyser/src/metrics/bandit_metrics/plugins/undefined_severity_issues.py

from .base import BanditMetricPlugin


# 🧠 ML Signal: Missing severity fields indicate potential tool or integration failures
# ⚠️ SAST Risk: Unclassified vulnerabilities may be overlooked in dashboards or prioritisation
# ✅ Best Practice: Tag with 'undefined' and expose via overlay system
class UndefinedSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues with unknown or missing severity.

    Returns:
        int: Count of issues without standard severity labels.
    """

    plugin_name = "number_of_undefined_security_vulnerabilities"
    plugin_tags = ["security", "bandit", "severity", "undefined", "incomplete"]

    def name(self) -> str:
        return self.plugin_name

    def extract(self, data: dict) -> int:
        known = {"LOW", "MEDIUM", "HIGH"}
        return sum(
            1
            for item in data.get("results", [])
            if item.get("issue_severity", "").strip().upper() not in known
        )

    def confidence_score(self, data: dict) -> float:
        count = self.extract(data)
        return min(1.0, 0.2 + 0.2 * count)

    def severity_level(self, data: dict) -> str:
        count = self.extract(data)
        if count == 0:
            return "low"
        elif count <= 2:
            return "medium"
        else:
            return "high"
