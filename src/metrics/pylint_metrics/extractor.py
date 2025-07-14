import json
import logging
import subprocess
from typing import Dict, Any
from metrics.pylint_metrics.plugins import load_plugins

logger = logging.getLogger(__name__)

class PylintMetricExtractor:
    """
    Extracts Pylint metrics by running pylint and applying plugin-based analysis.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.plugins = load_plugins()
        self.result_metrics: Dict[str, Any] = {}

    def extract(self) -> Dict[str, Any]:
        """
        Run pylint and compute metrics using plugins.

        Returns:
            dict[str, int | float]: Computed pylint metrics.
        """
        try:
            result = subprocess.run(
                ["pylint", "--output-format=json", self.file_path],
                capture_output=True,
                text=True,
                check=False
            )
        except Exception as e:
            logger.error(f"[PylintMetricExtractor] Failed to run pylint on {self.file_path}: {type(e).__name__}: {e}")
            return self._default_metrics()

        if not result.stdout.strip():
            logger.warning(f"[PylintMetricExtractor] No output from pylint on {self.file_path}")
            return self._default_metrics()

        try:
            messages = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.error(f"[PylintMetricExtractor] JSON parse error on {self.file_path}: {e}")
            return self._default_metrics()

        for plugin in self.plugins:
            try:
                value = plugin.extract(messages, self.file_path)
                self.result_metrics[plugin.name()] = value if isinstance(value, (int, float)) else 0
            except Exception as e:
                logger.warning(f"[PylintMetricExtractor] Plugin {plugin.name()} failed: {type(e).__name__}: {e}")
                self.result_metrics[plugin.name()] = 0

        self._log_metrics()
        return self.result_metrics

    def _default_metrics(self) -> Dict[str, Any]:
        """
        Return zeroed metrics in case of failure.

        Returns:
            dict[str, int | float]: Fallback values.
        """
        return {plugin.name(): 0 for plugin in self.plugins}

    def _log_metrics(self) -> None:
        """
        Log all extracted pylint metrics.
        """
        if not self.result_metrics:
            logger.info(f"[PylintMetricExtractor] No metrics computed for {self.file_path}")
            return
        lines = [f"{k}: {v}" for k, v in self.result_metrics.items()]
        logger.info(f"[PylintMetricExtractor] Metrics for {self.file_path}:\n" + "\n".join(lines))
