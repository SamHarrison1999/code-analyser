import logging
import subprocess
from typing import Union, List
from metrics.metric_types import MetricExtractorBase
from metrics.pyflakes_metrics.plugins.base import PyflakesMetricPlugin
from metrics.pyflakes_metrics.plugins import load_plugins

# 🧠 ML Signal: Plugin-based extraction allows extensible, labelled metrics
# ⚠️ SAST Risk: Parsing raw linter output may be brittle; sanitise carefully

class PyflakesExtractor(MetricExtractorBase):
    """
    Extracts static code issues using Pyflakes with plugin-based extensibility.
    """

    def __init__(self, file_path: str, plugins: List[PyflakesMetricPlugin] = None):
        self.file_path = file_path
        self.plugins = plugins if plugins is not None else load_plugins()
        self.result_metrics: dict[str, Union[int, float]] = {}

    def extract(self) -> dict[str, Union[int, float]]:
        """
        Runs Pyflakes on the given file and applies plugins to compute metrics.

        Returns:
            dict[str, int | float]: Dictionary of computed metrics.
        """
        try:
            result = subprocess.run(
                ["pyflakes", self.file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                check=False,
            )
            output_lines = result.stdout.splitlines()
        except Exception as e:
            logging.error(f"[PyflakesExtractor] Error running Pyflakes on {self.file_path}: {type(e).__name__}: {e}")
            return self._default_metrics()

        metrics = {}
        for plugin in self.plugins:
            try:
                value = plugin.extract(output_lines, self.file_path)
                metrics[plugin.name()] = value if isinstance(value, (int, float)) else 0
            except Exception as e:
                logging.warning(f"[PyflakesExtractor] Plugin '{plugin.name()}' failed: {type(e).__name__}: {e}")
                metrics[plugin.name()] = 0

        self.result_metrics = metrics
        self._log_metrics()
        return metrics

    def _default_metrics(self) -> dict[str, Union[int, float]]:
        """
        Provides a fallback metric dictionary if analysis fails.

        Returns:
            dict[str, int | float]: Zeroed plugin metrics.
        """
        return {plugin.name(): 0 for plugin in self.plugins}

    def _log_metrics(self):
        """
        Log the extracted Pyflakes metrics for debugging.
        """
        if not self.result_metrics:
            logging.info(f"[PyflakesExtractor] No metrics extracted for {self.file_path}")
            return

        lines = [f"{k}: {v}" for k, v in self.result_metrics.items()]
        logging.info(f"[PyflakesExtractor] Metrics for {self.file_path}:\n" + "\n".join(lines))


def extract_pyflakes_metrics(file_path: str) -> dict[str, Union[int, float]]:
    """
    Wrapper to extract metrics using the default plugin set.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        dict[str, int | float]: Extracted Pyflakes metrics.
    """
    return PyflakesExtractor(file_path).extract()
