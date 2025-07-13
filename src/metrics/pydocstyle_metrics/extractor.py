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
        Run pydocstyle on the target file and compute summary metrics.

        Returns:
            dict[str, int | float]: Metric dictionary containing:
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
            return self._default_metrics()

        violations = [line for line in output.splitlines() if line.strip()]
        total_violations = len(violations)
        missing_docstrings = sum(1 for line in violations if "Missing docstring" in line)
        try:
            compliance = round(((total_violations - missing_docstrings) / total_violations) * 100, 2) if total_violations else 100.0
        except ZeroDivisionError:
            compliance = 100.0

        metrics = {
            "number_of_pydocstyle_violations": total_violations,
            "number_of_missing_doc_strings": missing_docstrings,
            "percentage_of_compliance_with_docstring_style": compliance
        }

        logging.info(f"[PydocstyleExtractor] Metrics for {self.file_path}:\n{metrics}")
        return metrics

    def _default_metrics(self) -> dict[str, Union[int, float]]:
        """
        Provides a fallback result in case of execution failure.

        Returns:
            dict[str, int | float]: Zeroed/default metrics.
        """
        return {
            "number_of_pydocstyle_violations": 0,
            "number_of_missing_doc_strings": 0,
            "percentage_of_compliance_with_docstring_style": 0.0
        }
