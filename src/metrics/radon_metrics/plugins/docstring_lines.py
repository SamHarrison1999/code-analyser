from .base import RadonMetricPlugin


class DocstringLinesPlugin(RadonMetricPlugin):
    """
    Extracts the number of docstring lines reported by Radon.

    This metric helps assess documentation density and coverage within the codebase.
    """

    plugin_name = "docstring_lines"
    plugin_tags = ["radon", "documentation", "structure"]

    def name(self) -> str:
        """
        Returns:
            str: The metric name used in structured outputs.
        """
        return self.plugin_name

    # ✅ Updated to accept both 'parsed' and 'file_path' as required
    def extract(self, parsed: dict, file_path: str) -> int:
        """
        Extracts the number of docstring lines from Radon output.

        Args:
            parsed (dict): Parsed Radon output for a file.
            file_path (str): Path to the analysed source file.

        Returns:
            int: Count of docstring lines.
        """
        return parsed.get("docstring_lines", 0)

    def confidence_score(self, parsed: dict) -> float:
        """
        Returns:
            float: Confidence score of the metric based on data presence.
        """
        return 1.0 if "docstring_lines" in parsed else 0.0

    def severity_level(self, parsed: dict) -> str:
        """
        Returns:
            str: Severity classification based on docstring density.
        """
        count = parsed.get("docstring_lines", 0)
        return "low" if count > 5 else "medium" if count > 0 else "high"
