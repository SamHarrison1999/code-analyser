# File: code_analyser/src/metrics/vulture_metrics/extractor.py

"""
Initialise the Vulture metric extractor module.

Exposes key classes and plugin-based functionality for integration with the core runner.
"""

import logging
from typing import Any, Union, Dict, List
from vulture import Vulture
from metrics.metric_types import MetricExtractorBase
from metrics.vulture_metrics.plugins import load_plugins, VultureMetricPlugin

logger = logging.getLogger(__name__)


# 🧠 ML Signal: Plugin-based metrics enable label-rich scoring per unused-code dimension
# ⚠️ SAST Risk: Vulture parsing failures or malformed ASTs may impact analysis
class VultureMetricExtractor(MetricExtractorBase):
    """
    Extracts unused code metrics using Vulture and applies plugin-based metric extraction.

    Each plugin receives Vulture unused item list and computes a specific scalar metric.
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.plugins: List[VultureMetricPlugin] = load_plugins()
        self.data: List[Any] = []
        self.result_metrics: Dict[str, Union[int, float]] = {}

    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Run Vulture and extract all plugin-defined metrics.

        Returns:
            Dict[str, int | float]: Computed plugin results.
        """
        self.data = self._run_vulture()

        if not self.data:
            logger.warning(
                f"[VultureMetricExtractor] No results parsed for {self.file_path}"
            )
            return self._fallback_metrics()

        for plugin in self.plugins:
            try:
                value = plugin.extract(self.data)
                self.result_metrics[plugin.name()] = (
                    value if isinstance(value, (int, float)) else 0
                )
            except Exception as e:
                logger.warning(
                    f"[VultureMetricExtractor] Plugin '{plugin.name()}' failed: {type(e).__name__}: {e}"
                )
                self.result_metrics[plugin.name()] = 0

        self._log_metrics()
        return self.result_metrics

    def _run_vulture(self) -> List[Any]:
        """
        Run Vulture analysis and return raw unused item list.

        Returns:
            List[Any]: List of Vulture unused code items.
        """
        try:
            with open(self.file_path, encoding="utf-8") as f:
                code = f.read()

            v = Vulture()
            v.scan(code)
            return v.get_unused_code()

        except Exception as e:
            logger.warning(
                f"[VultureMetricExtractor] Vulture failed on {self.file_path}: {type(e).__name__}: {e}"
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
                f"[VultureMetricExtractor] No metrics extracted for {self.file_path}"
            )
            return

        lines = [f"{name}: {value}" for name, value in self.result_metrics.items()]
        logger.info(
            f"[VultureMetricExtractor] Metrics for {self.file_path}:\n"
            + "\n".join(lines)
        )


def extract_vulture_metrics(file_path: str) -> List[Dict[str, Union[str, float, int]]]:
    """
    Structured export of Vulture plugin metrics with confidence and severity.

    Args:
        file_path (str): Python file to analyse.

    Returns:
        List[Dict[str, object]]: Metric bundle for each plugin.
    """
    try:
        extractor = VultureMetricExtractor(file_path)
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
            f"[extract_vulture_metrics] Failed for {file_path}: {type(e).__name__}: {e}"
        )
        return [
            {"metric": plugin.name(), "value": 0, "confidence": 0.0, "severity": "low"}
            for plugin in load_plugins()
        ]


def get_vulture_extractor():
    """
    Returns the plugin-compatible extractor class for Vulture.

    Returns:
        type[VultureMetricExtractor]: Class reference.
    """
    return VultureMetricExtractor
