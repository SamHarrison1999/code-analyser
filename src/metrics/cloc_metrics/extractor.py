import json
import subprocess
import logging
from typing import Dict, Union
from metrics.cloc_metrics.plugins import load_plugins

class ClocExtractor:
    """
    Extracts and processes CLOC metrics via plugin architecture.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.plugins = load_plugins()
        self.raw_cloc_data: Dict = {}
        self.result_metrics: Dict[str, Union[int, float]] = {}

    def extract(self) -> Dict[str, Union[int, float]]:
        cloc_data = self._run_cloc()
        if not cloc_data:
            self.result_metrics = {p.name(): 0 for p in self.plugins}
            return self.result_metrics

        self.raw_cloc_data = cloc_data
        self._apply_plugins()
        return self.result_metrics

    def _run_cloc(self) -> Dict:
        try:
            proc = subprocess.run(
                ["cloc", "--json", self.file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                check=False
            )
            return json.loads(proc.stdout)
        except Exception as e:
            logging.warning(f"[ClocExtractor] Failed to run cloc: {e}")
            return {}

    def _apply_plugins(self):
        python_data = self.raw_cloc_data.get("Python", {})
        for plugin in self.plugins:
            try:
                val = plugin.extract(python_data)
                self.result_metrics[plugin.name()] = val if isinstance(val, (int, float)) else 0
            except Exception as e:
                logging.warning(f"[ClocExtractor] Plugin {plugin.name()} failed: {e}")
                self.result_metrics[plugin.name()] = 0
