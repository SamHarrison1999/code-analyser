from .base import SonarMetricPlugin


class VulnerabilitiesPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return "vulnerabilities"

    def extract(self, sonar_data: dict, file_path: str) -> float:
        """
        Extracts the 'vulnerabilities' metric from SonarQube analysis results.

        Args:
            sonar_data (dict): Parsed dictionary from SonarQube API.
            file_path (str): Path to the analysed file (unused).

        Returns:
            float: The number of reported vulnerabilities, or 0.0 if unavailable.
        """
        value = sonar_data.get("vulnerabilities")
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
