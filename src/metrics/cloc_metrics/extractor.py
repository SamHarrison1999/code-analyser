import json
import logging
import subprocess
from typing import Dict, Any, List

from metrics.cloc_metrics.plugins import load_plugins
from metrics.cloc_metrics.plugins.base import CLOCMetricPlugin


class ClocExtractor:
    """
    Extracts line-based metrics from a Python file using the `cloc` command.

    Metrics include comment density, total lines, source lines, and more,
    using a plugin-based architecture for modularity.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.plugins: List[BaseClocMetricPlugin] = load_plugins()
        self.result_metrics: Dict[str, Any] = {}
        self.raw_data: Dict[str, Any] = {}

    def extract(self) -> Dict[str, Any]:
        """
        Runs `cloc` and applies all registered metric plugins.

        Returns:
            dict[str, int | float]: Dictionary of computed metrics.
        """
        cloc_json = self._run_cloc()
        if not cloc_json:
            # Fail-safe: return 0 or default values for all known plugins
            self.result_metrics = {plugin.name(): 0.0 if "density" in plugin.name() else 0 for plugin in self.plugins}
            return self.result_metrics

        self.raw_data = cloc_json.get("Python", {})
        for plugin in self.plugins:
            try:
                self.result_metrics[plugin.name()] = plugin.extract(self.raw_data)
            except Exception as e:
                logging.warning(f"[ClocExtractor] Plugin {plugin.name()} failed: {e}")
                self.result_metrics[plugin.name()] = 0.0 if "density" in plugin.name() else 0

        self._log_metrics()
        return self.result_metrics

    def _run_cloc(self) -> Dict[str, Any] | None:
        """Execute cloc and return parsed JSON output."""
        try:
            output = subprocess.check_output(
                ["cloc", "--json", "--include-lang=Python", self.file_path],
                encoding="utf-8",
                stderr=subprocess.DEVNULL
            )
            return json.loads(output)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logging.error(f"[ClocExtractor] cloc error for {self.file_path}: {e}")
            return None
        except FileNotFoundError:
            logging.error("[ClocExtractor] cloc is not installed or not in PATH.")
            return None

    def _log_metrics(self):
        """Log computed metrics for transparency."""
        lines = [f"{k}: {v}" for k, v in self.result_metrics.items()]
        logging.info(f"[ClocExtractor] Metrics for {self.file_path}:\n" + "\n".join(lines))


def gather_cloc_metrics(file_path: str) -> List[Any]:
    """
    Returns ordered cloc metrics for ML pipelines or CSV export.

    Args:
        file_path (str): Path to Python file.

    Returns:
        list[Any]: Ordered list of cloc metrics.
    """
    extractor = ClocExtractor(file_path)
    metrics = extractor.extract()

    # Fixed order for ML/CSV consistency
    return [
        metrics.get("number_of_comments", 0),
        metrics.get("number_of_lines", 0),
        metrics.get("number_of_source_lines_of_code", 0),
        metrics.get("comment_density", 0.0),
    ]
