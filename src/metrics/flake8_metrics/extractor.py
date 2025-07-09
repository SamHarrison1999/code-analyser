import logging
import subprocess
from typing import Any, Dict


class Flake8Extractor:
    """
    Extracts styling and formatting metrics using Flake8 static analysis.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.result_metrics: Dict[str, Any] = {}

    def extract(self) -> Dict[str, Any]:
        """
        Runs Flake8 and parses diagnostic codes into structured metrics.

        Returns:
            dict[str, int | float]: A dictionary of formatting-related metrics.
        """
        metrics: dict[str, Any] = {
            "number_of_unused_variables": 0,
            "number_of_unused_imports": 0,
            "number_of_inconsistent_indentations": 0,
            "number_of_trailing_whitespaces": 0,
            "number_of_long_lines": 0,
            "number_of_doc_string_issues": 0,
            "number_of_naming_issues": 0,
            "number_of_whitespace_issues": 0,
            "average_line_length": 0.0,
            "number_of_styling_warnings": 0,
            "number_of_styling_errors": 0,
            "number_of_styling_issues": 0,
        }

        try:
            result = subprocess.run(
                ["flake8", self.file_path],
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )

            for line in result.stdout.splitlines():
                parts = line.strip().split(":", 3)
                if len(parts) < 4:
                    continue
                code_message = parts[3].strip()
                tokens = code_message.split()
                if not tokens:
                    continue
                code = tokens[0]
                metrics["number_of_styling_issues"] += 1

                if code == "F841":
                    metrics["number_of_unused_variables"] += 1
                elif code == "F401":
                    metrics["number_of_unused_imports"] += 1
                elif code in {"E111", "E114"}:
                    metrics["number_of_inconsistent_indentations"] += 1
                elif code in {"W291", "W293"}:
                    metrics["number_of_trailing_whitespaces"] += 1
                elif code == "E501":
                    metrics["number_of_long_lines"] += 1
                elif code.startswith("D"):
                    metrics["number_of_doc_string_issues"] += 1
                elif code.startswith("N"):
                    metrics["number_of_naming_issues"] += 1
                elif code in {"E201", "E202", "E221"}:
                    metrics["number_of_whitespace_issues"] += 1

                if code.startswith("W"):
                    metrics["number_of_styling_warnings"] += 1
                elif code.startswith(("E", "F")):
                    metrics["number_of_styling_errors"] += 1

            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    total_chars = sum(len(line.rstrip("\n")) for line in lines)
                    metrics["average_line_length"] = round(total_chars / len(lines), 2) if lines else 0.0
            except Exception as e:
                logging.warning(f"[Flake8Extractor] Line length calculation failed: {e}")
                metrics["average_line_length"] = 0.0

            self.result_metrics = metrics
            self._log_metrics()
            return self.result_metrics

        except Exception as e:
            logging.error(f"[Flake8Extractor] Unexpected error for {self.file_path}: {e}")
            return {key: 0 for key in metrics}

    def _log_metrics(self):
        lines = [f"{k}: {v}" for k, v in self.result_metrics.items()]
        logging.info(f"[Flake8Extractor] Metrics for {self.file_path}:\n" + "\n".join(lines))


def gather_flake8_metrics(file_path: str) -> list[Any]:
    """
    Extract Flake8 metrics in ordered list for ML/CSV export.
    """
    extractor = Flake8Extractor(file_path)
    metrics = extractor.extract()

    return [
        metrics.get("number_of_unused_variables", 0),
        metrics.get("number_of_unused_imports", 0),
        metrics.get("number_of_inconsistent_indentations", 0),
        metrics.get("number_of_trailing_whitespaces", 0),
        metrics.get("number_of_long_lines", 0),
        metrics.get("number_of_doc_string_issues", 0),
        metrics.get("number_of_naming_issues", 0),
        metrics.get("number_of_whitespace_issues", 0),
        metrics.get("average_line_length", 0.0),
        metrics.get("number_of_styling_warnings", 0),
        metrics.get("number_of_styling_errors", 0),
        metrics.get("number_of_styling_issues", 0),
    ]
