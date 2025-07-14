import json
import subprocess
import logging
from typing import Dict, Union, Any
from metrics.cloc_metrics.plugins import load_plugins

# ✅ Best Practice: Use plugin registry for tool-specific extractors
# ⚠️ SAST Risk: External tool invocation (cloc) can fail silently or return malformed output
# 🧠 ML Signal: Plugin extraction patterns can be used to learn which code structures affect metric generation

class ClocExtractor:
    """
    Extracts and processes CLOC metrics via plugin architecture.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.plugins = load_plugins()
        self.raw_cloc_data: Dict[str, Any] = {}
        self.result_metrics: Dict[str, Union[int, float]] = {}

    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Run CLOC and apply metric plugins to the parsed JSON result.

        Returns:
            A dictionary of plugin name to computed metric value.
        """
        cloc_data = self._run_cloc()
        if not cloc_data:
            # 🧠 ML Signal: When cloc output is empty, plugin fallback zeros are assigned
            self.result_metrics = {plugin.name(): 0 for plugin in self.plugins}
            return self.result_metrics

        self.raw_cloc_data = cloc_data
        self._apply_plugins()
        return self.result_metrics

    def _run_cloc(self) -> Dict[str, Any]:
        """
        Execute the `cloc` tool and parse the JSON output.

        Returns:
            Parsed dictionary from CLOC output or empty dict on failure.
        """
        try:
            proc = subprocess.run(
                ["cloc", "--json", self.file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                check=False
            )

            if proc.returncode > 1 or not proc.stdout:
                logging.warning(f"[ClocExtractor] cloc failed or returned no output for {self.file_path}")
                return {}

            return json.loads(proc.stdout)

        except json.JSONDecodeError as e:
            logging.warning(f"[ClocExtractor] Failed to decode JSON for {self.file_path}: {e}")
            return {}
        except Exception as e:
            logging.warning(f"[ClocExtractor] Unexpected error running cloc: {e}")
            return {}

    def _apply_plugins(self):
        """
        Run each plugin extractor on the Python block of cloc data.
        """
        python_data = self.raw_cloc_data.get("Python", {})
        for plugin in self.plugins:
            try:
                value = plugin.extract(python_data)
                if isinstance(value, (int, float)):
                    self.result_metrics[plugin.name()] = value
                else:
                    self.result_metrics[plugin.name()] = 0
            except Exception as e:
                logging.warning(f"[ClocExtractor] Plugin {plugin.name()} failed: {e}")
                self.result_metrics[plugin.name()] = 0
