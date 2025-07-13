from .base import BanditMetricPlugin

class MediumSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues with MEDIUM severity.

    Returns:
        int: Count of issues marked as MEDIUM severity.
    """
    def name(self) -> str:
        return "number_of_medium_security_vulnerabilities"

    def extract(self, data: dict) -> int:
        return sum(
            1
            for item in data.get("results", [])
            if item.get("issue_severity", "").strip().upper() == "MEDIUM"
        )
