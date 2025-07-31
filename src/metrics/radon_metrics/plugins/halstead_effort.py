from .base import RadonMetricPlugin


class HalsteadEffortPlugin(RadonMetricPlugin):
    """
    Computes the Halstead Effort metric from Radon analysis results.

    Halstead Effort is an estimate of mental effort required to implement or understand a module.
    Higher values indicate more complex code in terms of information processing.
    """

    plugin_name = "halstead_effort"
    plugin_tags = ["halstead", "complexity", "effort", "radon"]

    def name(self) -> str:
        """
        Return the unique identifier for this metric plugin.

        Returns:
            str: The metric name used as a key.
        """
        return self.plugin_name

    def extract(self, parsed: dict, file_path: str) -> float:
        """
        Extracts and returns the Halstead Effort metric from Radon output.

        Args:
            parsed (dict): Dictionary containing parsed Radon results.
            file_path (str): Path to the file being analysed (unused here).

        Returns:
            float: Halstead Effort score (rounded to 2 decimals).
        """
        return round(parsed.get("halstead_effort", 0.0), 2)

    def confidence_score(self, parsed: dict) -> float:
        """
        Returns a confidence score based on metric presence.

        Returns:
            float: 1.0 if present, 0.0 otherwise.
        """
        return 1.0 if "halstead_effort" in parsed else 0.0

    def severity_level(self, parsed: dict) -> str:
        """
        Returns severity level based on thresholding the effort score.

        Returns:
            str: 'low', 'medium', or 'high'.
        """
        effort = parsed.get("halstead_effort", 0.0)
        if effort > 5000:
            return "high"
        elif effort > 2000:
            return "medium"
        return "low"
