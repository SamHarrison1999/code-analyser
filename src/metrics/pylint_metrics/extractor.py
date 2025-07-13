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
            logger.error(f"[PylintMetricExtractor] Failed to run pylint: {e}")
            return {plugin.name(): 0 for plugin in self.plugins}

        if not result.stdout.strip():
            logger.warning(f"[PylintMetricExtractor] No output from pylint on {self.file_path}")
            return {plugin.name(): 0 for plugin in self.plugins}

        try:
            messages = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.error(f"[PylintMetricExtractor] JSON parse error: {e}")
            return {plugin.name(): 0 for plugin in self.plugins}

        metrics = {}
        for plugin in self.plugins:
            try:
                value = plugin.extract(messages, self.file_path)
                metrics[plugin.name()] = value
            except Exception as e:
                logger.warning(f"[PylintMetricExtractor] Plugin {plugin.name()} failed: {e}")
                metrics[plugin.name()] = 0

        return metrics
