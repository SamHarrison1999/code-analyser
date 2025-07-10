# File: metrics/pydocstyle_metrics/extractor.py

import logging
import subprocess
from typing import Union
from metrics.metric_types import MetricExtractorBase

class PydocstyleExtractor(MetricExtractorBase):
    """
    Extracts documentation style metrics from Python source files using pydocstyle.
    """

    def extract(self) -> dict[str, Union[int, float]]:
        try:
            result = subprocess.run(
                ["pydocstyle", self.file_path],
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False
            )

            if result.returncode == 0:
                metrics = {
                    "number_of_pydocstyle_violations": 0,
                    "number_of_missing_doc_strings": 0,
                    "percentage_of_compliance_with_docstring_style": 100.0
                }
                logging.info(f"[PydocstyleExtractor] Metrics for {self.file_path}:\n{metrics}")
                return metrics

            output = result.stdout

        except Exception as e:
            logging.error(f"[PydocstyleExtractor] Error running pydocstyle on {self.file_path}: {e}")
            output = ""

        try:
            violations = output.splitlines() if output else []
            total = len(violations)
            missing_docstrings = sum(1 for line in violations if "Missing docstring" in line)
            compliance_percentage = 100.0 if total == 0 else ((total - missing_docstrings) / total) * 100

            metrics = {
                "number_of_pydocstyle_violations": total,
                "number_of_missing_doc_strings": missing_docstrings,
                "percentage_of_compliance_with_docstring_style": round(compliance_percentage, 2)
            }

            logging.info(f"[PydocstyleExtractor] Metrics for {self.file_path}:\n{metrics}")
            return metrics

        except Exception as ex:
            logging.error(f"[PydocstyleExtractor] Failed to parse pydocstyle output for {self.file_path}: {ex}")
            return {
                "number_of_pydocstyle_violations": 0,
                "number_of_missing_doc_strings": 0,
                "percentage_of_compliance_with_docstring_style": 0.0
            }
