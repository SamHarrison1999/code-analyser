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
            logging.error(f"[PyflakesExtractor] Error running Pyflakes on {self.file_path}: {e}")
            return {plugin.name(): 0 for plugin in self.plugins}

        metrics = {}
        for plugin in self.plugins:
            try:
                metrics[plugin.name()] = plugin.extract(output_lines, self.file_path)
            except Exception as e:
                logging.warning(f"[PyflakesExtractor] Plugin '{plugin.name()}' failed: {e}")
                metrics[plugin.name()] = 0

        self.result_metrics = metrics
        logging.info(f"[PyflakesExtractor] Metrics for {self.file_path}:\n{metrics}")
        return metrics


def extract_pyflakes_metrics(file_path: str) -> dict[str, Union[int, float]]:
    """
    Wrapper to extract metrics using the default plugin set.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        dict[str, int | float]: Extracted Pyflakes metrics.
    """
    return PyflakesExtractor(file_path).extract()
