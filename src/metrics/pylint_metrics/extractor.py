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
        # Store absolute path to source file for consistent subprocess calls
        self.file_path = os.path.abspath(file_path)
        logging.debug(f"[PylintMetricExtractor] Initialized with file: {self.file_path}")

    def _get_pylint_executable(self) -> str:
        """
        Returns the absolute path to pylint executable,
        resolving bundled pylint.exe when frozen (e.g., PyInstaller).
        Falls back to system pylint or 'pylint' command.
        """
        if getattr(sys, "frozen", False):
            base_path = getattr(sys, "_MEIPASS", None)
            if base_path:
                candidate = os.path.join(base_path, "pylint.exe")
                if os.path.isfile(candidate):
                    logging.debug(f"[PylintMetricExtractor] Using bundled pylint at: {candidate}")
                    return candidate
            logging.debug("[PylintMetricExtractor] sys._MEIPASS not found or pylint.exe missing")
        pylint_path = shutil.which("pylint")
        if pylint_path:
            logging.debug(f"[PylintMetricExtractor] Using system pylint at: {pylint_path}")
            return pylint_path
        logging.warning("[PylintMetricExtractor] pylint executable not found, using 'pylint'")
        return "pylint"

    def extract(self) -> Dict[str, int]:
        """
        Run Pylint on the target file and aggregate issue counts
        by severity type (convention, refactor, warning, error, fatal).

        Returns:
            Dict[str, int]: Metrics dictionary with keys prefixed by 'pylint_'.
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
                logging.debug(f"[PylintMetricExtractor] stderr: {result.stderr.strip()}")

            output = result.stdout.strip()
            if not output:
                logging.debug("[PylintMetricExtractor] No output from pylint, returning zeros")
                return self._empty_metrics()

            messages = json.loads(output)
            counter = {k: 0 for k in ["convention", "refactor", "warning", "error", "fatal"]}
            for msg in messages:
                msg_type = msg.get("type")
                if msg_type in counter:
                    counter[msg_type] += 1

            logging.debug(f"[PylintMetricExtractor] Metrics extracted: {counter}")
            return {f"pylint_{k}": v for k, v in counter.items()}

        except Exception as e:
            logging.error(f"[PylintMetricExtractor] Exception during extraction: {e}")
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict[str, int]:
        # Returns zeroed metrics for all pylint severity types
        return {f"pylint_{k}": 0 for k in ["convention", "refactor", "warning", "error", "fatal"]}
