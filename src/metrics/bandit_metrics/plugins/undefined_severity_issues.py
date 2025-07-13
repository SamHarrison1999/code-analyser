from .base import BanditMetricPlugin

class UndefinedSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues with unknown or missing severity.

    Returns:
        int: Count of issues without standard severity labels.
    """
    def name(self) -> str:
        return "number_of_undefined_security_vulnerabilities"

    def extract(self, data: dict) -> int:
        known = {"LOW", "MEDIUM", "HIGH"}
        return sum(
            1
            for item in data.get("results", [])
            if item.get("issue_severity", "").strip().upper() not in known
        )
