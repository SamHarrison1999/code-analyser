# File: code_analyser/src/metrics/pydocstyle_metrics/extractor.py

"""
Initialise the Pydocstyle metric extractor module.

Exposes key classes and functions for integration with the main metric runner.
"""

import logging
import subprocess
from typing import List, Dict, Union
from metrics.metric_types import MetricExtractorBase
from metrics.pydocstyle_metrics.plugins import load_plugins, PydocstyleMetricPlugin

logger = logging.getLogger(__name__)


class PydocstyleExtractor(MetricExtractorBase):
    """
    Extracts docstring compliance metrics using Pydocstyle via plugin system.

    Each plugin consumes raw Pydocstyle diagnostic lines and computes one metric.
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.plugins: List[PydocstyleMetricPlugin] = load_plugins()
        self.data: List[str] = []  # Raw Pydocstyle output
        self.result_metrics: Dict[str, Union[int, float]] = {}

    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Run Pydocstyle and apply plugin-based metric extraction.

        Returns:
            Dict[str, int | float]: Dictionary of plugin metric outputs.
        """
        try:
            self.data = self._run_pydocstyle()

            if not self.data:
                logger.debug(
                    f"[PydocstyleExtractor] No Pydocstyle output for {self.file_path}"
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
                        f"[PydocstyleExtractor] Plugin '{plugin.name()}' failed: {type(e).__name__}: {e}"
                    )
                    self.result_metrics[plugin.name()] = 0

            self._log_metrics()
            return self.result_metrics

        except Exception as e:
            logger.error(
                f"[PydocstyleExtractor] Failed to analyse {self.file_path}: {type(e).__name__}: {e}"
            )
            return self._fallback_metrics()

    def _run_pydocstyle(self) -> List[str]:
        """
        Execute the `pydocstyle` tool on the file and return diagnostic lines.

        Returns:
            List[str]: Parsed Pydocstyle output lines.
        """
        try:
            proc = subprocess.run(
                ["pydocstyle", self.file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                check=False,
            )
            if proc.returncode not in (0, 1):  # 1 = violations found
                logger.warning(
                    f"[PydocstyleExtractor] Pydocstyle returned exit code {proc.returncode} for {self.file_path}"
                )
                return []
            return proc.stdout.strip().splitlines()
        except Exception as e:
            logger.error(
                f"[PydocstyleExtractor] Subprocess failed: {type(e).__name__}: {e}"
            )
            return []

    def _fallback_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Provides fallback zero values for all plugin metrics.

        Returns:
            Dict[str, int | float]: Zero-valued metric dictionary.
        """
        return {plugin.name(): 0 for plugin in self.plugins}

    def _log_metrics(self):
        """
        Logs the plugin-derived metric results.
        """
        if not self.result_metrics:
            logger.info(
                f"[PydocstyleExtractor] No metrics extracted for {self.file_path}"
            )
            return
        lines = [f"{name}: {value}" for name, value in self.result_metrics.items()]
        logger.info(
            f"[PydocstyleExtractor] Metrics for {self.file_path}:\n" + "\n".join(lines)
        )


def extract_pydocstyle_metrics(
    file_path: str,
) -> List[Dict[str, Union[str, float, int]]]:
    """
    Extract plugin-based Pydocstyle metrics with confidence and severity.

    Args:
        file_path (str): Python file to analyse.

    Returns:
        List[Dict[str, Any]]: Bundle of plugin output and metadata.
    """
    try:
        extractor = PydocstyleExtractor(file_path)
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
        logger.warning(
            f"[extract_pydocstyle_metrics] Bundle extraction failed for {file_path}: {type(e).__name__}: {e}"
        )
        return [
            {"metric": plugin.name(), "value": 0, "confidence": 0.0, "severity": "low"}
            for plugin in load_plugins()
        ]


def get_pydocstyle_extractor():
    """
    Return the extractor class for plugin integration or runtime dispatch.

    Returns:
        type[PydocstyleExtractor]: Reference to the extractor class.
    """
    return PydocstyleExtractor
