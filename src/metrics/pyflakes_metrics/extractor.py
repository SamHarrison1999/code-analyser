# File: code_analyser/src/metrics/pyflakes_metrics/extractor.py

"""
Initialise the Pyflakes metric extractor module.

Exposes key classes and plugin-based functionality for integration with the core runner.
"""

import logging
import subprocess
from typing import Union, Dict, List
from metrics.metric_types import MetricExtractorBase
from metrics.pyflakes_metrics.plugins import load_plugins, PyflakesMetricPlugin


# 🧠 ML Signal: Plugin-based extraction allows extensible, labelled metrics
# ⚠️ SAST Risk: Parsing raw linter output may be brittle; sanitise carefully
class PyflakesExtractor(MetricExtractorBase):
    """
    Extracts static code diagnostics using Pyflakes via plugin-based architecture.

    Each plugin receives parsed output lines and computes a scalar metric.
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.plugins: List[PyflakesMetricPlugin] = load_plugins()
        self.data: List[str] = []  # ✅ Used for plugin scoring
        self.result_metrics: Dict[str, Union[int, float]] = {}

    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Run Pyflakes and apply plugin extractors.

        Returns:
            Dict[str, int | float]: Plugin-computed metric results.
        """
        self.data = self._run_pyflakes()

        if not self.data:
            logging.warning(
                f"[PyflakesExtractor] No diagnostics produced by Pyflakes for {self.file_path}"
            )
            return self._fallback_metrics()

        for plugin in self.plugins:
            try:
                value = plugin.extract(self.data, self.file_path)
                self.result_metrics[plugin.name()] = value if isinstance(value, (int, float)) else 0
            except Exception as e:
                logging.warning(
                    f"[PyflakesExtractor] Plugin '{plugin.name()}' failed: {type(e).__name__}: {e}"
                )
                self.result_metrics[plugin.name()] = 0

        self._log_metrics()
        return self.result_metrics

    def _run_pyflakes(self) -> List[str]:
        """
        Run Pyflakes on the file and return output lines.

        Returns:
            List[str]: Diagnostic lines emitted by Pyflakes.
        """
        try:
            result = subprocess.run(
                ["pyflakes", self.file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                check=False,
            )
            return result.stdout.splitlines()
        except Exception as e:
            logging.error(f"[PyflakesExtractor] Error running Pyflakes: {type(e).__name__}: {e}")
            return []

    def _fallback_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Zero-valued fallback metric dictionary for all plugins.

        Returns:
            Dict[str, int | float]: Default values on failure.
        """
        return {plugin.name(): 0 for plugin in self.plugins}

    def _log_metrics(self):
        """
        Print final extracted metrics for traceability.
        """
        if not self.result_metrics:
            logging.info(f"[PyflakesExtractor] No metrics extracted for {self.file_path}")
            return

        lines = [f"{k}: {v}" for k, v in self.result_metrics.items()]
        logging.info(f"[PyflakesExtractor] Metrics for {self.file_path}:\n" + "\n".join(lines))


def extract_pyflakes_metrics(file_path: str) -> List[Dict[str, Union[str, float, int]]]:
    """
    Structured wrapper for bundle-style export of Pyflakes plugin metrics.

    Args:
        file_path (str): Path to analyse.

    Returns:
        List[Dict[str, object]]: Each dict includes metric name, value, confidence, severity.
    """
    try:
        extractor = PyflakesExtractor(file_path)
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
        logging.warning(
            f"[extract_pyflakes_metrics] Failed for {file_path}: {type(e).__name__}: {e}"
        )
        return [
            {"metric": plugin.name(), "value": 0, "confidence": 0.0, "severity": "low"}
            for plugin in load_plugins()
        ]


def get_pyflakes_extractor():
    """
    Return the extractor class for plugin-based metric runners.

    Returns:
        type[PyflakesExtractor]: The extractor class.
    """
    return PyflakesExtractor
