from .base import SonarMetricPlugin


class SecurityRatingPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return "security_rating"

    def extract(self, sonar_data: dict, file_path: str) -> float:
        """
        Extracts the 'security_rating' metric from the SonarQube analysis data.

        Args:
            sonar_data (dict): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used directly).

        Returns:
            float: The security rating, or 0.0 if missing or invalid.
        """
        value = sonar_data.get("security_rating")
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
