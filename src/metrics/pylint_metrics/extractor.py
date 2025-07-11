import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List


class PylintMetricExtractor:
    """
    Extracts Pylint-based static metrics from a Python source file.

    This extractor runs Pylint as a subprocess and collects structured
    diagnostics that can be used for ML, auditing, or code scoring.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.metrics: List[Dict[str, Any]] = []

    def extract(self) -> List[Dict[str, Any]]:
        """
        Executes Pylint and returns a list of structured metrics.

        Each entry includes:
        - type: category (e.g. 'convention', 'error')
        - symbol: rule name (e.g. 'missing-docstring')
        - message: descriptive text
        - line, column: position in file (if available)
        - message_id: short ID (e.g. 'C0114')
        """
        try:
            result = subprocess.run(
                ["pylint", "--output-format=json", str(self.file_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
            )
            output = result.stdout.strip()

            if not output:
                return []

            messages = json.loads(output)
            self.metrics = [
                {
                    "type": msg.get("type"),
                    "symbol": msg.get("symbol"),
                    "message": msg.get("message"),
                    "line": msg.get("line"),
                    "column": msg.get("column"),
                    "message_id": msg.get("message-id"),
                }
                for msg in messages
            ]
        except Exception as e:
            logging.error(f"[PylintExtractor] Failed to extract metrics: {e}")
            self.metrics = []

        return self.metrics


def gather_pylint_metrics(file_path: Path) -> List[Dict[str, Any]]:
    """
    Convenience wrapper to extract pylint metrics using the PylintMetricExtractor.
    """
    extractor = PylintMetricExtractor(str(file_path))
    return extractor.extract()
