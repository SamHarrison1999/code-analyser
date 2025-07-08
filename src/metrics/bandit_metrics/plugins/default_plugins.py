# File: src/metrics/bandit_metrics/plugins/default_plugins.py

from .base import BanditMetricPlugin


class HighSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues with HIGH severity.
    """

    def name(self) -> str:
        return "number_of_high_security_vulnerabilities"

    def extract(self, data: dict) -> int:
        return sum(
            1 for item in data.get("results", [])
            if item.get("issue_severity", "").upper() == "HIGH"
        )


class MediumSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues with MEDIUM severity.
    """

    def name(self) -> str:
        return "number_of_medium_security_vulnerabilities"

    def extract(self, data: dict) -> int:
        return sum(
            1 for item in data.get("results", [])
            if item.get("issue_severity", "").upper() == "MEDIUM"
        )


class LowSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues with LOW severity.
    """

    def name(self) -> str:
        return "number_of_low_security_vulnerabilities"

    def extract(self, data: dict) -> int:
        return sum(
            1 for item in data.get("results", [])
            if item.get("issue_severity", "").upper() == "LOW"
        )


class UndefinedSeverityIssues(BanditMetricPlugin):
    """
    Counts the number of Bandit-reported issues that have no defined severity
    or use a non-standard label.
    """

    def name(self) -> str:
        return "number_of_undefined_security_vulnerabilities"

    def extract(self, data: dict) -> int:
        valid = {"LOW", "MEDIUM", "HIGH"}
        return sum(
            1 for item in data.get("results", [])
            if item.get("issue_severity", "").upper() not in valid
        )
