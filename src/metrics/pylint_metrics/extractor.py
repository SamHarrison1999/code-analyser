import sys
import os
import shutil
import subprocess
import json
import logging
from typing import Dict


class PylintMetricExtractor:
    """
    Extracts summary metrics from Pylint output as a flat dictionary
    of message type counts: convention, refactor, warning, error, fatal.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = os.path.abspath(file_path)
        logging.debug(f"[PylintMetricExtractor] Initialized with file: {self.file_path}")

    def _get_pylint_executable(self) -> str:
        """
        Returns the absolute path to the pylint executable.

        - If frozen (e.g. PyInstaller), attempts to locate bundled pylint.exe.
        - Otherwise, uses system-installed pylint.
        """
        if getattr(sys, "frozen", False):
            base_path = getattr(sys, "_MEIPASS", "")
            candidate = os.path.join(base_path, "pylint.exe")
            if os.path.isfile(candidate):
                logging.debug(f"[PylintMetricExtractor] Using bundled pylint at: {candidate}")
                return candidate
            logging.debug("[PylintMetricExtractor] Bundled pylint.exe not found in _MEIPASS")

        system_path = shutil.which("pylint")
        if system_path:
            logging.debug(f"[PylintMetricExtractor] Using system pylint at: {system_path}")
            return system_path

        logging.warning("[PylintMetricExtractor] No pylint executable found, falling back to 'pylint'")
        return "pylint"

    def extract(self) -> Dict[str, int]:
        """
        Runs pylint on the file and returns severity type counts as a dictionary.

        Returns:
            Dict[str, int]: A dict with keys: convention, refactor, warning, error, fatal.
        """
        try:
            pylint_exe = self._get_pylint_executable()
            cmd = [pylint_exe, "--output-format=json", self.file_path]
            logging.debug(f"[PylintMetricExtractor] Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if result.stderr:
                logging.warning(f"[PylintMetricExtractor] stderr output:\n{result.stderr.strip()}")

            output = result.stdout.strip()
            if not output:
                logging.warning("[PylintMetricExtractor] No pylint output found. Returning empty metrics.")
                return self._empty_metrics()

            try:
                messages = json.loads(output)
            except json.JSONDecodeError as e:
                logging.error(f"[PylintMetricExtractor] JSON parse error: {e}")
                return self._empty_metrics()

            counts = {k: 0 for k in ["convention", "refactor", "warning", "error", "fatal"]}
            for msg in messages:
                msg_type = msg.get("type")
                if msg_type in counts:
                    counts[msg_type] += 1

            logging.debug(f"[PylintMetricExtractor] Extracted metrics: {counts}")
            return counts

        except Exception as e:
            logging.exception(f"[PylintMetricExtractor] Unexpected error: {e}")
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict[str, int]:
        """
        Return zeroed default metrics.
        """
        return {k: 0 for k in ["convention", "refactor", "warning", "error", "fatal"]}
