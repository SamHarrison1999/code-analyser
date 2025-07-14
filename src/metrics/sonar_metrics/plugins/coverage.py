from .base import SonarMetricPlugin


class CoveragePlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return "coverage"

    def extract(self, sonar_data: dict, file_path: str) -> float:
        """
        Extracts the 'coverage' metric from SonarQube analysis output.

        Args:
            sonar_data (dict): Dictionary of parsed SonarQube results.
            file_path (str): Path to the analysed file (unused).

        Returns:
            float: The code coverage percentage, or 0.0 if unavailable or invalid.
        """
        value = sonar_data.get("coverage")
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
