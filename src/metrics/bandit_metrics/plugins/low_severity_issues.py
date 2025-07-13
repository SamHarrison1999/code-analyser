from .base import BanditMetricPlugin

class LowSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues with LOW severity.

    Returns:
        int: Count of issues marked as LOW severity.
    """
    def name(self) -> str:
        return "number_of_low_security_vulnerabilities"

    def extract(self, data: dict) -> int:
        return sum(
            1
            for item in data.get("results", [])
            if item.get("issue_severity", "").strip().upper() == "LOW"
        )
