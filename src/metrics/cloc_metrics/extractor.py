import json
import logging
import subprocess
from typing import Dict, Any


class ClocExtractor:
    """
    Extracts line-based metrics from a Python file using the `cloc` command.
    Metrics include comment density, line counts, and source lines of code.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.result_metrics: Dict[str, Any] = {}

    def extract(self) -> Dict[str, Any]:
        """
        Runs `cloc` and returns a dictionary of metrics.

        Returns:
            dict[str, int | float]: Metrics about lines, comments, and density.
        """
        try:
            output = subprocess.check_output(
                ["cloc", "--json", self.file_path],
                encoding="utf-8"
            )
            data = json.loads(output)
            metrics = data.get("Python", {})

            blank = metrics.get("blank", 0)
            comment = metrics.get("comment", 0)
            code = metrics.get("code", 0)
            total_lines = blank + comment + code
            comment_density = comment / total_lines if total_lines > 0 else 0.0

            self.result_metrics = {
                "number_of_comments": comment,
                "number_of_lines": total_lines,
                "number_of_source_lines_of_code": code,
                "comment_density": round(comment_density, 4)
            }

            self._log_metrics()
            return self.result_metrics

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logging.error(f"[ClocExtractor] cloc error for {self.file_path}: {e}")
            return {
                "number_of_comments": 0,
                "number_of_lines": 0,
                "number_of_source_lines_of_code": 0,
                "comment_density": 0.0
            }

    def _log_metrics(self):
        """Logs the computed metrics for traceability."""
        lines = [f"{k}: {v}" for k, v in self.result_metrics.items()]
        logging.info(f"[ClocExtractor] Metrics for {self.file_path}:\n" + "\n".join(lines))


def gather_cloc_metrics(file_path: str) -> list[Any]:
    """
    Gathers ordered cloc metrics for use in CSV/ML output.

    Args:
        file_path (str): Path to Python source file.

    Returns:
        list[Any]: Ordered values of cloc metrics.
    """
    extractor = ClocExtractor(file_path)
    metrics = extractor.extract()

    return [
        metrics.get("number_of_comments", 0),
        metrics.get("number_of_lines", 0),
        metrics.get("number_of_source_lines_of_code", 0),
        metrics.get("comment_density", 0.0),
    ]
