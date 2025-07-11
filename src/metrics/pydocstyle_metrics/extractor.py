"""
PydocstyleExtractor: Extracts docstring style metrics using the pydocstyle tool.
"""

import logging
import subprocess
from typing import Union
from metrics.metric_types import MetricExtractorBase


class PydocstyleExtractor(MetricExtractorBase):
    """
    Extracts documentation style metrics from Python source files using pydocstyle.
    """

    def extract(self) -> dict[str, Union[int, float]]:
        """
        Runs pydocstyle on the target file and parses the output
        to compute docstring compliance metrics.

        Returns:
            dict[str, int | float]: Metric dictionary including:
                - number_of_pydocstyle_violations
                - number_of_missing_doc_strings
                - percentage_of_compliance_with_docstring_style
        """
        try:
            result = subprocess.run(
                ["pydocstyle", self.file_path],
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False
            )
            output = result.stdout or ""
        except Exception as e:
            logging.error(f"[PydocstyleExtractor] Error running pydocstyle on {self.file_path}: {e}")
            return {
                "number_of_pydocstyle_violations": 0,
                "number_of_missing_doc_strings": 0,
                "percentage_of_compliance_with_docstring_style": 0.0
            }

        # Split output into diagnostic lines
        violations = output.splitlines()
        total = len(violations)
        missing_docstrings = sum(1 for line in violations if "Missing docstring" in line)
        compliance = 100.0 if total == 0 else round(((total - missing_docstrings) / total) * 100, 2)

        metrics = {
            "number_of_pydocstyle_violations": total,
            "number_of_missing_doc_strings": missing_docstrings,
            "percentage_of_compliance_with_docstring_style": compliance
        }

        logging.info(f"[PydocstyleExtractor] Metrics for {self.file_path}:\n{metrics}")
        return metrics
