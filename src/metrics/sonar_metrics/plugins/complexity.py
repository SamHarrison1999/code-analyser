from .base import SonarMetricPlugin


class ComplexityPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return "complexity"

    def extract(self, sonar_data: dict, file_path: str) -> float:
        """
        Extract the 'complexity' metric from SonarQube analysis output.

        Args:
            sonar_data (dict): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used by this plugin).

        Returns:
            float: The total code complexity value, or 0.0 if unavailable or invalid.
        """
        value = sonar_data.get("complexity")
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
