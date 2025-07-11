# File: metrics/pyflakes_metrics/extractor.py

import logging
import subprocess
from typing import Union
from metrics.metric_types import MetricExtractorBase


class PyflakesExtractor(MetricExtractorBase):
    """
    Extracts static code issues using pyflakes.
    """

    def extract(self) -> dict[str, Union[int, float]]:
        """
        Runs pyflakes on the given file and parses basic issue counts.

        Returns:
            dict[str, int | float]: Dictionary containing:
                - number_of_undefined_names
                - number_of_syntax_errors
                - number_of_pyflakes_issues
        """
        try:
            result = subprocess.run(
                ["pyflakes", self.file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                check=False
            )
            output = result.stdout.splitlines()
        except Exception as e:
            logging.error(f"[PyflakesExtractor] Error running pyflakes on {self.file_path}: {e}")
            return {
                "number_of_undefined_names": 0,
                "number_of_syntax_errors": 0,
                "number_of_pyflakes_issues": 0
            }

        # 🔍 Count lines containing specific error types
        num_undefined = sum("undefined name" in line for line in output)
        num_syntax = sum("syntax error" in line.lower() for line in output)

        metrics = {
            "number_of_undefined_names": num_undefined,
            "number_of_syntax_errors": num_syntax,
            "number_of_pyflakes_issues": len(output)
        }

        logging.info(f"[PyflakesExtractor] Metrics for {self.file_path}:\n{metrics}")
        return metrics


# ✅ Required by unified CLI entry point
def extract_pyflakes_metrics(file_path: str) -> dict[str, Union[int, float]]:
    """
    Convenience wrapper for CLI and plugin integration.

    Args:
        file_path (str): Path to Python file to analyse.

    Returns:
        dict[str, int | float]: Extracted Pyflakes metrics.
    """
    return PyflakesExtractor(file_path).extract()
