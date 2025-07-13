import logging
import subprocess
from typing import Any, Dict, List

from metrics.flake8_metrics.plugins import load_plugins
from metrics.flake8_metrics.plugins.base import Flake8MetricPlugin

logger = logging.getLogger(__name__)


class Flake8Extractor:
    """
    Extracts style and formatting metrics using Flake8 static analysis,
    then computes results using dynamically loaded plugin metrics.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.plugins: List[Flake8MetricPlugin] = load_plugins()
        self.result_metrics: Dict[str, Any] = {}

    def extract(self) -> Dict[str, Any]:
        """
        Run Flake8 on the given file and extract plugin-based metrics.

        Returns:
            Dict[str, int | float]: Metric values keyed by plugin name.
        """
        try:
            result = subprocess.run(
                ["flake8", self.file_path],
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            flake8_output = result.stdout.splitlines()

            metrics = {}
            for plugin in self.plugins:
                try:
                    # ✅ Match plugin API: extract(flake8_output: List[str], file_path: str)
                    value = plugin.extract(flake8_output, self.file_path)
                    metrics[plugin.name()] = value
                except Exception as e:
                    logger.warning(f"[Flake8Extractor] Plugin '{plugin.name()}' failed: {type(e).__name__}: {e}")
                    metrics[plugin.name()] = 0

            self.result_metrics = metrics
            self._log_metrics()
            return metrics

        except Exception as e:
            logger.error(f"[Flake8Extractor] Flake8 execution failed on {self.file_path}: {type(e).__name__}: {e}")
            return {plugin.name(): 0 for plugin in self.plugins}

    def _log_metrics(self):
        """Log the extracted metrics for debugging and traceability."""
        lines = [f"{k}: {v}" for k, v in self.result_metrics.items()]
        logger.info(f"[Flake8Extractor] Metrics for {self.file_path}:\n" + "\n".join(lines))


def gather_flake8_metrics(file_path: str) -> List[Any]:
    """
    Extracts Flake8 metrics as an ordered list for ML or CSV output.

    Args:
        file_path (str): Path to the Python file to analyse.

    Returns:
        List[Any]: Ordered list of Flake8 metric values.
    """
    extractor = Flake8Extractor(file_path)
    metric_dict = extractor.extract()
    plugin_order = [plugin.name() for plugin in extractor.plugins]
    return [metric_dict.get(name, 0) for name in plugin_order]
