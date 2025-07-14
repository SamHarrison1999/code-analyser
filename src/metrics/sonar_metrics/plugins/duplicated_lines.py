from .base import SonarMetricPlugin


class DuplicatedLinesPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique metric name provided by this plugin.

        Returns:
            str: Metric name used in the result dictionary.
        """
        return "duplicated_lines"

    def extract(self, sonar_data: dict, file_path: str) -> float:
        """
        Extracts the 'duplicated_lines' metric from SonarQube results.

        Args:
            sonar_data (dict): Parsed SonarQube analysis result.
            file_path (str): Path to the analysed file (unused in this plugin).

        Returns:
            float: Number of duplicated lines, or 0.0 if missing or invalid.
        """
        value = sonar_data.get("duplicated_lines")
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
