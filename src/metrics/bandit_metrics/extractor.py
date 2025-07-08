# File: src/metrics/bandit_metrics/extractor.py

import json
import logging
import subprocess
from typing import Dict, List, Any

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
        self.result_metrics: Dict[str, int] = {}

    def extract(self) -> Dict[str, int]:
        """
        Runs Bandit and applies plugins to compute metrics.

        Returns:
            dict[str, int]: A dictionary of computed metric values.
        """
        bandit_data = self._run_bandit()
        if bandit_data is None:
            self.result_metrics = {p.name(): 0 for p in self.plugins}
            return self.result_metrics

        self.raw_bandit_data = bandit_data
        self._apply_plugins()
        self._log_metrics()
        return self.result_metrics

    def _run_bandit(self) -> Dict[str, Any] | None:
        """Run Bandit on the target file and return parsed JSON."""
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
        """Apply each plugin to compute its metric."""
        for plugin in self.plugins:
            try:
                self.result_metrics[plugin.name()] = plugin.extract(self.raw_bandit_data)
            except Exception as e:
                logging.warning(f"[BanditExtractor] Plugin {plugin.name()} failed: {e}")
                self.result_metrics[plugin.name()] = 0

    def _log_metrics(self):
        """Log the final computed metrics."""
        lines = [f"{name}: {val}" for name, val in self.result_metrics.items()]
        logging.info(f"[BanditExtractor] Metrics for {self.file_path}:\n" + "\n".join(lines))


def gather_bandit_metrics(file_path: str) -> List[int]:
    """
    Extract Bandit plugin metrics as an ordered list for ML/CSV export.

    Args:
        file_path (str): Path to the Python file to scan.

    Returns:
        list[int]: Plugin metrics in registered order.
    """
    extractor = BanditExtractor(file_path)
    metrics = extractor.extract()

    # ✅ Best Practice: Maintain plugin ordering for column-aligned outputs
    # 🧠 ML Signal: Column stability ensures retraining and comparisons are meaningful
    plugin_order = [p.name() for p in extractor.plugins]
    return [metrics.get(name, 0) for name in plugin_order]
