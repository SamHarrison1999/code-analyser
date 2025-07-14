from .base import SonarMetricPlugin


class DuplicatedBlocksPlugin(SonarMetricPlugin):
    def name(self) -> str:
        """
        Returns the unique metric name provided by this plugin.

        Returns:
            str: Metric name used in the result dictionary.
        """
        return "duplicated_blocks"

    def extract(self, sonar_data: dict, file_path: str) -> float:
        """
        Extracts the 'duplicated_blocks' metric from SonarQube results.

        Args:
            sonar_data (dict): SonarQube API data for the analysed file.
            file_path (str): Path to the target file (unused).

        Returns:
            float: Number of duplicated blocks, or 0.0 if unavailable or invalid.
        """
        value = sonar_data.get("duplicated_blocks")
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
