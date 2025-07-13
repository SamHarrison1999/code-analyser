"""
Initialise the Lizard metric extractor module.

Exposes key classes and functions for integration with the main metric runner.
"""

import logging
from typing import Union, Dict
from metrics.metric_types import MetricExtractorBase
from lizard import analyze_file


class LizardExtractor(MetricExtractorBase):
    """
    Extracts complexity and maintainability metrics using Lizard.

    Metrics include:
    - average_cyclomatic_complexity
    - average_token_count
    - total_function_count
    - max_cyclomatic_complexity
    - average_parameters
    """

    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Analyse the file using Lizard and extract summary metrics.

        Returns:
            Dict[str, int | float]: Dictionary of Lizard metrics.
        """
        try:
            analysis_result = analyze_file(self.file_path)
            functions = analysis_result.function_list

            if not functions:
                logging.debug(f"[LizardExtractor] No functions found in: {self.file_path}")
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
            logging.error(f"[LizardExtractor] Failed to analyse {self.file_path}: {type(e).__name__}: {e}")
            return self._default_metrics()

    def _default_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Provides zeroed fallback values when analysis fails.

        Returns:
            Dict[str, int | float]: Default zero-valued metric dictionary.
        """
        return {
            "average_cyclomatic_complexity": 0.0,
            "average_token_count": 0.0,
            "total_function_count": 0,
            "max_cyclomatic_complexity": 0,
            "average_parameters": 0.0,
        }


def extract_lizard_metrics(file_path: str) -> Dict[str, Union[int, float]]:
    """
    Extracts Lizard metrics using a convenience wrapper.

    Args:
        file_path (str): Path to the file to analyse.

    Returns:
        Dict[str, int | float]: Dictionary of extracted metrics.
    """
    return LizardExtractor(file_path).extract()


def get_lizard_extractor():
    """
    Returns the LizardExtractor class for plugin-compatible usage.

    Returns:
        type[LizardExtractor]: Extractor class reference.
    """
    return LizardExtractor
