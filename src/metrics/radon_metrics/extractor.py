import logging
from typing import Union
from radon.raw import analyze
from radon.metrics import h_visit
from metrics.metric_types import MetricExtractorBase

# ✅ Configure module logger
logger = logging.getLogger(__name__)


class RadonExtractor(MetricExtractorBase):
    """
    Extracts raw and Halstead complexity metrics using Radon.
    """

    def extract(self) -> dict[str, Union[int, float]]:
        """
        Computes a summary of raw and Halstead metrics from the file.

        Returns:
            dict[str, int | float]: Metric dictionary containing:
                - logical_lines
                - blank_lines
                - docstring_lines
                - halstead_volume
                - halstead_difficulty
                - halstead_effort
        """
        try:
            with open(self.file_path, encoding="utf-8") as f:
                code = f.read()

            raw_metrics = analyze(code)
            halstead = h_visit(code)

            metrics = {
                "logical_lines": raw_metrics.lloc,
                "blank_lines": raw_metrics.blank,
                "docstring_lines": raw_metrics.comments,
                "halstead_volume": round(halstead.total.volume, 2),
                "halstead_difficulty": round(halstead.total.difficulty, 2),
                "halstead_effort": round(halstead.total.effort, 2),
            }

            logger.info(f"[RadonExtractor] Metrics for {self.file_path}:\n{metrics}")
            return metrics

        except Exception as e:
            logger.warning(f"[RadonExtractor] Failed to analyse {self.file_path}: {type(e).__name__}: {e}")
            return self._default_metrics()

    def _default_metrics(self) -> dict[str, Union[int, float]]:
        """
        Fallback values if analysis fails.

        Returns:
            dict[str, int | float]: Zeroed or empty metrics.
        """
        return {
            "logical_lines": 0,
            "blank_lines": 0,
            "docstring_lines": 0,
            "halstead_volume": 0.0,
            "halstead_difficulty": 0.0,
            "halstead_effort": 0.0,
        }


def extract_radon_metrics(file_path: str) -> dict[str, Union[int, float]]:
    """
    Runs the Radon extractor and returns a dictionary of metrics.

    Args:
        file_path (str): The file to analyse.

    Returns:
        dict[str, int | float]: Computed metric results.
    """
    return RadonExtractor(file_path).extract()


__all__ = ["RadonExtractor", "extract_radon_metrics"]
