from .base import SonarMetricPlugin


class SqaleRatingPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return "sqale_rating"

    def extract(self, sonar_data: dict, file_path: str) -> float:
        """
        Extracts the 'sqale_rating' metric from the SonarQube analysis results.

        Args:
            sonar_data (dict): Dictionary returned by the SonarQube API.
            file_path (str): Path to the analysed source file (not used here).

        Returns:
            float: The technical debt rating, or 0.0 if unavailable or invalid.
        """
        value = sonar_data.get("sqale_rating")
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
