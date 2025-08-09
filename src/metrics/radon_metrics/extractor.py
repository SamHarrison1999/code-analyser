# File: code_analyser/src/metrics/radon_metrics/extractor.py

"""
Initialise the Radon metric extractor module.

Exposes key classes and plugin-based functionality for integration with the core runner.
"""

import logging
from typing import Union, Dict, List
from radon.raw import analyze
from radon.metrics import h_visit
from metrics.metric_types import MetricExtractorBase
from metrics.radon_metrics.plugins import load_plugins, RadonMetricPlugin

logger = logging.getLogger(__name__)


# 🧠 ML Signal: Plugin-based metrics enable label-rich scoring per Radon dimension
# ⚠️ SAST Risk: Radon parsing failures or missing AST nodes may affect downstream scoring
class RadonMetricExtractor(MetricExtractorBase):
    """
    Extracts raw metrics using Radon and applies plugin-based metric extraction.

    Each plugin receives Radon summary metrics and computes a specific scalar metric.
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.plugins: List[RadonMetricPlugin] = load_plugins()
        self.data: Dict[str, Union[int, float]] = {}
        self.result_metrics: Dict[str, Union[int, float]] = {}

    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Run Radon and extract all plugin-defined metrics.

        Returns:
            Dict[str, int | float]: Computed plugin results.
        """
        self.data = self._run_radon()

        if not self.data:
            logger.warning(f"[RadonMetricExtractor] No metrics parsed for {self.file_path}")
            return self._fallback_metrics()

        for plugin in self.plugins:
            try:
                value = plugin.extract(self.data, self.file_path)
                self.result_metrics[plugin.name()] = value if isinstance(value, (int, float)) else 0
            except Exception as e:
                logger.warning(
                    f"[RadonMetricExtractor] Plugin '{plugin.name()}' failed: {type(e).__name__}: {e}"
                )
                self.result_metrics[plugin.name()] = 0

        self._log_metrics()
        return self.result_metrics

    def _run_radon(self) -> Dict[str, Union[int, float]]:
        """
        Run Radon analysis and return parsed metrics.

        Returns:
            Dict[str, int | float]: Raw Radon metric summary.
        """
        try:
            with open(self.file_path, encoding="utf-8") as f:
                code = f.read()

            raw = analyze(code)
            halstead = h_visit(code)

            return {
                "logical_lines": raw.lloc,
                "blank_lines": raw.blank,
                "docstring_lines": raw.comments,
                "halstead_volume": round(halstead.total.volume, 2),
                "halstead_difficulty": round(halstead.total.difficulty, 2),
                "halstead_effort": round(halstead.total.effort, 2),
            }
        except Exception as e:
            logger.warning(
                f"[RadonMetricExtractor] Radon failed on {self.file_path}: {type(e).__name__}: {e}"
            )
            return {}

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
            logger.info(f"[RadonMetricExtractor] No metrics extracted for {self.file_path}")
            return

        lines = [f"{name}: {value}" for name, value in self.result_metrics.items()]
        logger.info(f"[RadonMetricExtractor] Metrics for {self.file_path}:\n" + "\n".join(lines))


def extract_radon_metrics(file_path: str) -> List[Dict[str, Union[str, float, int]]]:
    """
    Structured export of Radon plugin metrics with confidence and severity.

    Args:
        file_path (str): Python file to analyse.

    Returns:
        List[Dict[str, object]]: Metric bundle for each plugin.
    """
    try:
        extractor = RadonMetricExtractor(file_path)
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
        logger.warning(f"[extract_radon_metrics] Failed for {file_path}: {type(e).__name__}: {e}")
        return [
            {"metric": plugin.name(), "value": 0, "confidence": 0.0, "severity": "low"}
            for plugin in load_plugins()
        ]


def get_radon_extractor():
    """
    Returns the plugin-compatible extractor class for Radon.

    Returns:
        type[RadonMetricExtractor]: Class reference.
    """
    return RadonMetricExtractor
