# File: code_analyser/src/metrics/flake8_metrics/plugins/styling_errors_plugin.py

from .base import Flake8MetricPlugin
from typing import List


class StylingErrorPlugin(Flake8MetricPlugin):
    """
    Counts the number of Flake8 styling errors, specifically those with codes
    starting with 'E' (pycodestyle) or 'F' (pyflakes).
    """

    # ✅ Best Practice: Plugin metadata for registry and filtering
    plugin_name = "number_of_styling_errors"
    plugin_tags = ["style", "E", "F", "formatting", "pep8", "pyflakes"]

    def name(self) -> str:
        return self.plugin_name

    # ⚠️ SAST Risk: Depends on consistent Flake8 output including 'E'/'F' prefixed codes
    # 🧠 ML Signal: Styling violations are strong signals for automated formatting need
    def extract(self, flake8_output: List[str], file_path: str) -> int:
        """
        Counts the number of Flake8 styling errors related to error codes starting with 'E' or 'F'.

        Args:
            flake8_output (List[str]): List of Flake8 output lines.
            file_path (str): Path to the analysed file (unused in this plugin).

        Returns:
            int: Count of styling errors (E* or F* codes).
        """
        return sum(
            1 for line in flake8_output if any(f" {prefix}" in line for prefix in ("E", "F"))
        )

    # ✅ Best Practice: Confidence is high if output is non-empty
    def confidence_score(self, flake8_output: List[str]) -> float:
        return 1.0 if flake8_output else 0.0

    # ✅ Best Practice: Severity determined by styling error volume
    def severity_level(self, flake8_output: List[str]) -> str:
        count = self.extract(flake8_output, file_path="")
        if count >= 20:
            return "high"
        elif count >= 5:
            return "medium"
        return "low"
