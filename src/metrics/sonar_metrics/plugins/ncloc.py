from .base import SonarMetricPlugin


class NclocPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique name of the metric this plugin provides.

        Returns:
            str: The metric identifier used in result dictionaries.
        """
        return "ncloc"

    def extract(self, sonar_data: dict, file_path: str) -> float:
        """
        Extracts the 'ncloc' (non-comment lines of code) metric from the SonarQube data.

        Args:
            sonar_data (dict): Parsed results from SonarQube API.
            file_path (str): Path to the analysed file (not used directly).

        Returns:
            float: The value of ncloc or 0.0 if unavailable or invalid.
        """
        value = sonar_data.get("ncloc")
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
