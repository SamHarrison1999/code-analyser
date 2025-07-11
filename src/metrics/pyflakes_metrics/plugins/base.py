# File: metrics/pyflakes_metrics/plugins/base.py

"""
Base structure for Pyflakes metric plugins.

This module defines a standard interface for implementing Pyflakes-related
plugins in a structured and extensible way.
"""

from abc import ABC, abstractmethod


class PyflakesPlugin(ABC):
    """
    Abstract base class for Pyflakes plugins.

    All plugins must implement the `extract` method which takes the Pyflakes output
    and file path, returning a dictionary of metric names and values.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Returns the name of the metric this plugin extracts.

        Returns:
            str: The name of the metric.
        """
        pass

    @abstractmethod
    def extract(self, pyflakes_output: list[str], file_path: str) -> int:
        """
        Extracts the metric value from the given Pyflakes output.

        Args:
            pyflakes_output (list[str]): The list of Pyflakes warnings and errors.
            file_path (str): Path to the analysed Python file.

        Returns:
            int: Metric value extracted from the Pyflakes output.
        """
        pass
