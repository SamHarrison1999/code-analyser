# File: code_analyser/src/metrics/lizard_metrics/extractor.py

"""
Initialise the Lizard metric extractor module.

Exposes key classes and functions for integration with the main metric runner.
"""

import logging
from typing import Union, Dict, Any, List
from metrics.metric_types import MetricExtractorBase
from metrics.lizard_metrics.plugins import load_plugins, LizardMetricPlugin
from lizard import analyze_file


class LizardExtractor(MetricExtractorBase):
    """
    Extracts complexity and maintainability metrics using Lizard via plugin system.

    Each plugin consumes parsed Lizard function-level data and produces one metric.
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.plugins: List[LizardMetricPlugin] = load_plugins()
        self.data: List[Dict[str, Any]] = []  # ✅ Used by plugin confidence/severity
        self.result_metrics: Dict[str, Union[int, float]] = {}

    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Analyse the file using Lizard and apply plugin-based metric extraction.

        Returns:
            Dict[str, int | float]: Dictionary of Lizard metric outputs.
        """
        try:
            result = analyze_file(self.file_path)
            self.data = self._prepare_lizard_entries(result.function_list)

            if not self.data:
                logging.debug(f"[LizardExtractor] No valid function entries in: {self.file_path}")
                return self._fallback_metrics()

            for plugin in self.plugins:
                try:
                    value = plugin.extract(self.data, self.file_path)
                    self.result_metrics[plugin.name()] = (
                        value if isinstance(value, (int, float)) else 0
                    )
                except Exception as e:
                    logging.warning(
                        f"[LizardExtractor] Plugin '{plugin.name()}' failed: {type(e).__name__}: {e}"
                    )
                    self.result_metrics[plugin.name()] = 0

            self._log_metrics()
            return self.result_metrics

        except Exception as e:
            logging.error(
                f"[LizardExtractor] Failed to analyse {self.file_path}: {type(e).__name__}: {e}"
            )
            return self._fallback_metrics()

    def _prepare_lizard_entries(self, function_list) -> List[Dict[str, Any]]:
        """
        Normalises Lizard function data into plugin-compatible entries.

        Returns:
            List[Dict[str, Any]]: Parsed Lizard metrics for plugin use.
        """
        if not function_list:
            return []

        return (
            [
                {
                    "name": "average_cyclomatic_complexity",
                    "value": f.cyclomatic_complexity,
                }
                for f in function_list
            ]
            + [{"name": "average_token_count", "value": f.token_count} for f in function_list]
            + [
                {"name": "average_parameter_count", "value": len(f.parameters)}
                for f in function_list
            ]
            + [
                {"name": "max_cyclomatic_complexity", "value": f.cyclomatic_complexity}
                for f in function_list
            ]
            + [{"name": "number_of_functions", "value": 1} for _ in function_list]
        )

    def _fallback_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Provides fallback values for all registered plugins on failure.

        Returns:
            Dict[str, int | float]: Zero-valued results.
        """
        return {plugin.name(): 0 for plugin in self.plugins}

    def _log_metrics(self):
        """
        Logs the plugin-derived metrics.
        """
        if not self.result_metrics:
            logging.info(f"[LizardExtractor] No metrics extracted for {self.file_path}")
            return

        lines = [f"{name}: {value}" for name, value in self.result_metrics.items()]
        logging.info(f"[LizardExtractor] Metrics for {self.file_path}:\n" + "\n".join(lines))


def extract_lizard_metrics(file_path: str) -> List[Dict[str, Union[str, float, int]]]:
    """
    Extracts Lizard plugin metrics including value, confidence, and severity.

    Args:
        file_path (str): Path to the Python file to analyse.

    Returns:
        List[Dict[str, object]]: Structured plugin output bundle, e.g.
        [
            {"metric": "average_function_length", "value": 12.5, "confidence": 1.0, "severity": "medium"},
            ...
        ]
    """
    try:
        extractor = LizardExtractor(file_path)
        metrics = extractor.extract()

        return [
            {
                "metric": plugin.name(),
                "value": metrics.get(plugin.name(), 0),
                "confidence": round(plugin.confidence_score(extractor.data), 2),
                "severity": plugin.severity_level(extractor.data),
            }
            for plugin in extractor.plugins
        ]
    except Exception as e:
        logging.warning(
            f"[extract_lizard_metrics] Failed to extract bundle for {file_path}: {type(e).__name__}: {e}"
        )
        return [
            {"metric": plugin.name(), "value": 0, "confidence": 0.0, "severity": "low"}
            for plugin in load_plugins()
        ]


def get_lizard_extractor():
    """
    Returns the LizardExtractor class for dynamic loading.

    Returns:
        type[LizardExtractor]: The extractor class.
    """
    return LizardExtractor
