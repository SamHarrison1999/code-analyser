from .base import RadonMetricPlugin


class HalsteadVolumePlugin(RadonMetricPlugin):
    """
    Computes the Halstead Volume metric from Radon analysis results.

    Halstead Volume represents the size of a program in terms of information content.
    Higher values indicate more cognitive load due to program length and vocabulary size.
    """

    plugin_name = "halstead_volume"
    plugin_tags = ["halstead", "complexity", "volume", "radon"]

    def name(self) -> str:
        """
        Return the unique identifier for this metric plugin.

        Returns:
            str: The metric name used as a key.
        """
        return self.plugin_name

    def extract(self, parsed: dict, file_path: str) -> float:
        """
        Extracts and returns the Halstead Volume metric from Radon output.

        Args:
            parsed (dict): Dictionary containing parsed Radon results.
            file_path (str): Path to the file being analysed (unused here).

        Returns:
            float: Halstead Volume score (rounded to 2 decimals).
        """
        return round(parsed.get("halstead_volume", 0.0), 2)

    def confidence_score(self, parsed: dict) -> float:
        """
        Returns a confidence score based on metric presence.

        Returns:
            float: 1.0 if present, 0.0 otherwise.
        """
        return 1.0 if "halstead_volume" in parsed else 0.0

    def severity_level(self, parsed: dict) -> str:
        """
        Returns severity level based on thresholding the volume score.

        Returns:
            str: 'low', 'medium', or 'high'.
        """
        volume = parsed.get("halstead_volume", 0.0)
        if volume > 8000:
            return "high"
        elif volume > 4000:
            return "medium"
        return "low"
