from .base import SonarMetricPlugin


class TestSuccessDensityPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return "test_success_density"

    def extract(self, sonar_data: dict, file_path: str) -> float:
        """
        Extracts the 'test_success_density' metric from SonarQube analysis results.

        Args:
            sonar_data (dict): Parsed dictionary from SonarQube metrics API.
            file_path (str): Path to the analysed file (not used by this plugin).

        Returns:
            float: The success density of tests, or 0.0 if unavailable or invalid.
        """
        value = sonar_data.get("test_success_density")
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
