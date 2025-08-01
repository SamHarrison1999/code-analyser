# File: code_analyser/src/metrics/radon_metrics/plugins/logical_lines.py

from .base import RadonMetricPlugin


class LogicalLinesPlugin(RadonMetricPlugin):
    """
    Computes the number of logical lines of code from Radon output.

    Logical lines represent executable statements and are a core indicator
    of code size and potential complexity.
    """

    plugin_name = "logical_lines"
    plugin_tags = ["size", "lines", "radon", "structure"]

    def name(self) -> str:
        """
        Return the unique metric name.
        """
        return self.plugin_name

    def extract(self, parsed: dict, file_path: str) -> int:
        """
        Extracts the logical line count from parsed Radon data.

        Args:
            parsed (dict): Parsed Radon output.
            file_path (str): Path to file (not used in this metric).

        Returns:
            int: Number of logical lines of code.
        """
        return parsed.get("logical_lines", 0)

    def confidence_score(self, parsed: dict) -> float:
        """
        Always confident if metric key is present.

        Returns:
            float: Confidence score (0.0 to 1.0).
        """
        return 1.0 if "logical_lines" in parsed else 0.0

    def severity_level(self, parsed: dict) -> str:
        """
        Severity based on line count thresholds.

        Returns:
            str: 'low', 'medium', or 'high'.
        """
        lines = parsed.get("logical_lines", 0)
        if lines >= 500:
            return "high"
        elif lines >= 200:
            return "medium"
        return "low"
