"""
Initialise the Pylint metric extractor module.

Exposes key classes and plugin-based functionality for integration with the core runner.
"""

import json
import logging
import subprocess
from typing import Union, Dict, Any, List
from metrics.metric_types import MetricExtractorBase
from metrics.pylint_metrics.plugins import load_plugins, PylintMetricPlugin

logger = logging.getLogger(__name__)


# 🧠 ML Signal: Plugin-based metrics enable label-rich scoring per linter dimension
# ⚠️ SAST Risk: JSON parsing failures or unexpected diagnostic fields may corrupt metric interpretation
class PylintMetricExtractor(MetricExtractorBase):
    """
    Extracts diagnostics from Pylint and applies plugin-based metric extraction.

    Each plugin receives parsed Pylint JSON output and computes a specific scalar metric.
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.plugins: List[PylintMetricPlugin] = load_plugins()
        self.data: List[Dict[str, Any]] = []
        self.result_metrics: Dict[str, Union[int, float]] = {}

    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Run Pylint and extract all plugin-defined metrics.

        Returns:
            Dict[str, int | float]: Computed plugin results.
        """
        self.data = self._run_pylint()

        if not self.data:
            logger.warning(
                f"[PylintMetricExtractor] No diagnostics found for {self.file_path}"
            )
            return self._fallback_metrics()

        for plugin in self.plugins:
            try:
                value = plugin.extract(self.data, self.file_path)
                self.result_metrics[plugin.name()] = (
                    value if isinstance(value, (int, float)) else 0
                )
            except Exception as e:
                logger.warning(
                    f"[PylintMetricExtractor] Plugin '{plugin.name()}' failed: {type(e).__name__}: {e}"
                )
                self.result_metrics[plugin.name()] = 0

        self._log_metrics()
        return self.result_metrics

    def _run_pylint(self) -> List[Dict[str, Any]]:
        """
        Run Pylint and return its parsed JSON output.

        Returns:
            List[Dict[str, Any]]: List of diagnostic entries from Pylint.
        """
        try:
            result = subprocess.run(
                ["pylint", "--output-format=json", self.file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                check=False,
            )
            return json.loads(result.stdout.strip() or "[]")
        except json.JSONDecodeError as e:
            logger.error(f"[PylintMetricExtractor] JSON parse failed: {e}")
            return []
        except Exception as e:
            logger.error(
                f"[PylintMetricExtractor] Failed to run Pylint: {type(e).__name__}: {e}"
            )
            return []

    def _fallback_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Return fallback zeroed values for all plugins.

        Returns:
            Dict[str, int | float]: Plugin keys mapped to 0.
        """
        return {plugin.name(): 0 for plugin in self.plugins}

    def _log_metrics(self) -> None:
        """
        Log the final results after extraction.
        """
        if not self.result_metrics:
            logger.info(
                f"[PylintMetricExtractor] No metrics extracted for {self.file_path}"
            )
            return

        lines = [f"{name}: {value}" for name, value in self.result_metrics.items()]
        logger.info(
            f"[PylintMetricExtractor] Metrics for {self.file_path}:\n"
            + "\n".join(lines)
        )


def extract_pylint_metrics(file_path: str) -> List[Dict[str, Union[str, float, int]]]:
    """
    Structured export of Pylint plugin metrics with confidence and severity.

    Args:
        file_path (str): Python file to analyse.

    Returns:
        List[Dict[str, object]]: Metric bundle for each plugin.
    """
    try:
        extractor = PylintMetricExtractor(file_path)
        results = extractor.extract()

        return [
            {
                "metric": plugin.name(),
                "value": results.get(plugin.name(), 0),
                "confidence": round(plugin.confidence_score(extractor.data), 2),
                "severity": plugin.severity_level(extractor.data),
            }
            for plugin in extractor.plugins
        ]
    except Exception as e:
        logger.warning(
            f"[extract_pylint_metrics] Failed for {file_path}: {type(e).__name__}: {e}"
        )
        return [
            {"metric": plugin.name(), "value": 0, "confidence": 0.0, "severity": "low"}
            for plugin in load_plugins()
        ]


def get_pylint_extractor():
    """
    Returns the plugin-compatible extractor class for Pylint.

    Returns:
        type[PylintMetricExtractor]: Class reference.
    """
    return PylintMetricExtractor
