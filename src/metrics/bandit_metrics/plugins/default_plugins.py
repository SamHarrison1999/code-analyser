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
