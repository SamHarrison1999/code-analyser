"""
Default plugin implementations for Pylint metric extraction.

Each plugin analyses Pylint output and computes one metric.
"""

from typing import List, Dict, Any
from .base import PylintMetricPlugin


class MissingDocstringPlugin(PylintMetricPlugin):
    """
    Counts the number of missing docstring-related messages in Pylint output.
    Includes module, class, and function-level docstrings.
    """

    @staticmethod
    def name() -> str:
        return "missing_docstrings"

    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> int:
        return sum(
            1 for msg in pylint_output
            if msg.get("symbol") in {
                "missing-docstring",
                "missing-module-docstring",
                "missing-class-docstring",
                "missing-function-docstring",
            }
        )


class DuplicateCodePlugin(PylintMetricPlugin):
    """
    Counts the number of duplicate code messages in Pylint output.
    """

    @staticmethod
    def name() -> str:
        return "duplicate_code_blocks"

    def extract(self, pylint_output: List[Dict[str, Any]], file_path: str) -> int:
        return sum(
            1 for msg in pylint_output
            if msg.get("symbol") == "duplicate-code"
        )


DEFAULT_PLUGINS = [
    MissingDocstringPlugin,
    DuplicateCodePlugin,
]


def load_plugins() -> List[PylintMetricPlugin]:
    """
    Instantiate and return all default Pylint plugins.

    Returns:
        List[PylintMetricPlugin]: Instances of default plugins.
    """
    return [plugin() for plugin in DEFAULT_PLUGINS]


__all__ = [
    "MissingDocstringPlugin",
    "DuplicateCodePlugin",
    "DEFAULT_PLUGINS",
    "load_plugins",
]
