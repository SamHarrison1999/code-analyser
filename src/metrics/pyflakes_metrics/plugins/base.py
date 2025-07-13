"""
Base structure for Pyflakes metric plugins.

This module defines a standard interface for implementing Pyflakes-related
plugins in a structured and extensible way. Each plugin computes a specific
metric from the output of Pyflakes diagnostics.
"""

from abc import ABC, abstractmethod
from typing import List


class PyflakesMetricPlugin(ABC):
    """
    Abstract base class for Pyflakes plugins.

    All Pyflakes plugins must:
    - Provide a unique metric name via `name()`
    - Implement `extract()` to compute the metric from parsed output
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Returns:
            str: The unique name of the plugin metric.
        """
        raise NotImplementedError("PyflakesMetricPlugin must implement name() as a @classmethod.")

    @abstractmethod
    def extract(self, pyflakes_output: List[str], file_path: str) -> int:
        """
        Computes a metric from Pyflakes output.

        Args:
            pyflakes_output (List[str]): List of Pyflakes warnings and errors.
            file_path (str): Path to the source file being analysed.

        Returns:
            int: The computed metric value (e.g. count of a specific issue).
        """
        raise NotImplementedError("PyflakesMetricPlugin must implement extract().")
