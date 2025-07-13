from .base import BanditMetricPlugin

class HighSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues with HIGH severity.

    Returns:
        int: Count of issues marked as HIGH severity.
    """
    def name(self) -> str:
        return "number_of_high_security_vulnerabilities"

    def extract(self, data: dict) -> int:
        return sum(
            1
            for item in data.get("results", [])
            if item.get("issue_severity", "").strip().upper() == "HIGH"
        )
