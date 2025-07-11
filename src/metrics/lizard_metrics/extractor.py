"""
Initialise the Lizard metric extractor module.

Exposes key classes and functions for integration with the main metric runner.
"""

import logging
from typing import Union
from metrics.metric_types import MetricExtractorBase
from lizard import analyze_file


class LizardExtractor(MetricExtractorBase):
    """
    Extracts complexity and maintainability metrics using Lizard.
    """

    def extract(self) -> dict[str, Union[int, float]]:
        """
        Analyses the file using Lizard and extracts a summary of metrics.

        Returns:
            dict[str, int | float]: Dictionary containing:
                - average_cyclomatic_complexity
                - average_token_count
                - total_function_count
                - max_cyclomatic_complexity
                - average_parameters
        """
        try:
            analysis_result = analyze_file(self.file_path)
            functions = analysis_result.function_list

            if not functions:
                return self._default_metrics()

            complexities = [f.cyclomatic_complexity for f in functions]
            token_counts = [f.token_count for f in functions]
            parameter_counts = [len(f.parameters) for f in functions]

            metrics = {
                "average_cyclomatic_complexity": round(sum(complexities) / len(complexities), 2),
                "average_token_count": round(sum(token_counts) / len(token_counts), 2),
                "total_function_count": len(functions),
                "max_cyclomatic_complexity": max(complexities),
                "average_parameters": round(sum(parameter_counts) / len(parameter_counts), 2),
            }

            logging.info(f"[LizardExtractor] Metrics for {self.file_path}:\n{metrics}")
            return metrics

        except Exception as e:
            logging.error(f"[LizardExtractor] Error analysing {self.file_path}: {e}")
            return self._default_metrics()

    def _default_metrics(self) -> dict[str, Union[int, float]]:
        """
        Returns a default metrics dictionary with zeroed values.

        Returns:
            dict[str, int | float]: Zeroed fallback metrics.
        """
        return {
            "average_cyclomatic_complexity": 0.0,
            "average_token_count": 0.0,
            "total_function_count": 0,
            "max_cyclomatic_complexity": 0,
            "average_parameters": 0.0,
        }


def extract_lizard_metrics(file_path: str) -> dict[str, Union[int, float]]:
    """
    Convenience wrapper for CLI and plugin integration.

    Args:
        file_path (str): Path to Python file to analyse.

    Returns:
        dict[str, int | float]: Extracted Lizard metrics.
    """
    return LizardExtractor(file_path).extract()


def get_lizard_extractor():
    """
    Plugin-compatible entry point for Lizard metric extraction.

    Returns:
        type[LizardExtractor]: Callable extractor class for plugin runners.
    """
    return LizardExtractor
