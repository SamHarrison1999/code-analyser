# File: code_analyser/src/metrics/cloc_metrics/extractor.py

import json
import subprocess
import logging
from typing import Dict, Any, Union, Optional

# ✅ Best Practice: Use plugin namespace imports to avoid hardcoding deleted or renamed files
# ⚠️ SAST Risk: cloc output or subprocess may fail or produce malformed JSON
# 🧠 ML Signal: Plugin loading and result dictionary form input signals for downstream models

from metrics.cloc_metrics.plugins import load_plugins, ClocMetricPlugin


class ClocExtractor:
    """
    Extracts code size metrics using the CLOC utility and plugin-driven postprocessing.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.plugins: list[ClocMetricPlugin] = load_plugins()
        self.data: Dict[str, Any] = {}  # Parsed per-language data (e.g., cloc_data["Python"])
        self.result_metrics: Dict[str, Union[int, float]] = {}

    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Runs CLOC and applies plugins to compute metrics.

        Returns:
            dict[str, int | float]: A dictionary of computed metric values.
        """
        cloc_data = self._run_cloc()
        if cloc_data is None:
            # 🧠 ML Signal: Missing CLOC data implies fallback to zeros for each plugin
            self.result_metrics = {plugin.name(): 0 for plugin in self.plugins}
            return self.result_metrics

        self.data = cloc_data.get("Python", {})  # ✅ Best Practice: Extract Python-specific metrics
        self._apply_plugins()
        self._log_metrics()
        return self.result_metrics

    def _run_cloc(self) -> Optional[Dict[str, Any]]:
        """
        Run CLOC on the target file or directory and return parsed JSON.

        Returns:
            dict or None: Parsed CLOC output or None on failure.
        """
        try:
            proc = subprocess.run(
                ["cloc", "--json", self.file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                check=False,
            )

            if proc.returncode > 1:
                logging.warning(
                    f"[ClocExtractor] cloc failed with exit code {proc.returncode} for {self.file_path}"
                )
                return None

            return json.loads(proc.stdout)

        except json.JSONDecodeError as e:
            logging.warning(f"[ClocExtractor] JSON decode error for {self.file_path}: {e}")
            return None
        except Exception as e:
            logging.error(f"[ClocExtractor] Unexpected error running cloc: {e}")
            return None

    def _apply_plugins(self):
        """
        Apply each plugin to compute its metric with fallback on failure or bad data.
        """
        for plugin in self.plugins:
            try:
                value = plugin.extract(self.data)
                if isinstance(value, (int, float)):
                    self.result_metrics[plugin.name()] = value
                else:
                    logging.debug(
                        f"[ClocExtractor] Plugin {plugin.name()} returned non-numeric: {value!r}"
                    )
                    self.result_metrics[plugin.name()] = 0
            except Exception as e:
                logging.warning(f"[ClocExtractor] Plugin {plugin.name()} failed: {e}")
                self.result_metrics[plugin.name()] = 0

    def _log_metrics(self):
        """
        Log the final computed CLOC-based metrics.
        """
        if not self.result_metrics:
            logging.info(f"[ClocExtractor] No metrics computed for {self.file_path}")
            return

        lines = [f"{key}: {value}" for key, value in self.result_metrics.items()]
        logging.info(f"[ClocExtractor] Metrics for {self.file_path}:\n" + "\n".join(lines))
