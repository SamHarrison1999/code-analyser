from .base import SonarMetricPlugin


class ReliabilityRatingPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return "reliability_rating"

    def extract(self, sonar_data: dict, file_path: str) -> float:
        """
        Extracts the 'reliability_rating' metric from the SonarQube analysis data.

        Args:
            sonar_data (dict): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used directly).

        Returns:
            float: The reliability rating, or 0.0 if missing or invalid.
        """
        value = sonar_data.get("reliability_rating")
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
