from .base import SonarMetricPlugin


class DuplicatedLinesDensityPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique metric name provided by this plugin.

        Returns:
            str: Metric name used in the result dictionary.
        """
        return "duplicated_lines_density"

    def extract(self, sonar_data: dict, file_path: str) -> float:
        """
        Extracts the 'duplicated_lines_density' metric from SonarQube analysis results.

        Args:
            sonar_data (dict): Parsed SonarQube result dictionary.
            file_path (str): Path to the analysed file (not used here).

        Returns:
            float: Percentage of duplicated lines, or 0.0 if missing or invalid.
        """
        value = sonar_data.get("duplicated_lines_density")
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
