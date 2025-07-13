import json
import logging
import subprocess
from typing import Dict, Any, Union, Optional

# ✅ Best Practice: Use plugin namespace imports to avoid hardcoding deleted or renamed files
# ⚠️ SAST Risk: Static import of missing modules (like `default_plugins`) will crash the extractor pipeline
# 🧠 ML Signal: Plugin loading patterns inform which metrics are active and which tools require failover handling

from metrics.bandit_metrics.plugins import load_plugins

class BanditExtractor:
    """
    Extracts security vulnerability metrics using the Bandit static analysis tool.
    Uses plugin-driven postprocessing of Bandit JSON output.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.plugins = load_plugins()
        self.raw_bandit_data: Dict[str, Any] = {}
        self.result_metrics: Dict[str, Union[int, float]] = {}

    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Runs Bandit and applies plugins to compute metrics.

        Returns:
            dict[str, int | float]: A dictionary of computed metric values.
        """
        bandit_data = self._run_bandit()
        if bandit_data is None:
            self.result_metrics = {plugin.name(): 0 for plugin in self.plugins}
            return self.result_metrics

        self.raw_bandit_data = bandit_data
        self._apply_plugins()
        self._log_metrics()
        return self.result_metrics

    def _run_bandit(self) -> Optional[Dict[str, Any]]:
        """
        Run Bandit on the target file and return parsed JSON, or None on failure.
        """
        try:
            proc = subprocess.run(
                ["bandit", "-f", "json", "-r", self.file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                check=False,
            )
            if proc.returncode > 1:
                logging.warning(f"[BanditExtractor] Bandit failed with exit code {proc.returncode} for {self.file_path}")
                return None

            return json.loads(proc.stdout)

        except json.JSONDecodeError as e:
            logging.warning(f"[BanditExtractor] JSON parse error for {self.file_path}: {e}")
            return None
        except Exception as e:
            logging.error(f"[BanditExtractor] Unexpected error: {e}")
            return None

    def _apply_plugins(self):
        """
        Apply each plugin to compute its metric with numeric fallback on errors.
        """
        for plugin in self.plugins:
            try:
                value = plugin.extract(self.raw_bandit_data)
                if isinstance(value, (int, float)):
                    self.result_metrics[plugin.name()] = value
                else:
                    self.result_metrics[plugin.name()] = 0
            except Exception as e:
                logging.warning(f"[BanditExtractor] Plugin {plugin.name()} failed: {e}")
                self.result_metrics[plugin.name()] = 0

    def _log_metrics(self):
        """
        Log the final computed Bandit-based metrics.
        """
        if not self.result_metrics:
            logging.info(f"[BanditExtractor] No metrics computed for {self.file_path}")
            return

        lines = [f"{key}: {value}" for key, value in self.result_metrics.items()]
        logging.info(f"[BanditExtractor] Metrics for {self.file_path}:\n" + "\n".join(lines))
