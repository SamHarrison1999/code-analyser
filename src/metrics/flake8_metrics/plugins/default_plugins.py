"""
Default Flake8 metric plugins.

Each plugin extracts one specific metric from raw Flake8 output lines.

Provides:
- Plugin class definitions
- Plugin loader
- Ordered plugin list for ML/CSV export
"""

from typing import List
from metrics.flake8_metrics.plugins.base import Flake8Plugin


class UnusedVariablePlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_unused_variables"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        return sum(1 for line in flake8_output if ": F841 " in line)


class UnusedImportPlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_unused_imports"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        return sum(1 for line in flake8_output if ": F401 " in line)


class LongLinePlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_long_lines"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        return sum(1 for line in flake8_output if ": E501 " in line)


class DocstringIssuePlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_doc_string_issues"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        return sum(1 for line in flake8_output if ": D" in line)


class NamingIssuePlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_naming_issues"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        return sum(1 for line in flake8_output if ": N" in line)


class StylingErrorPlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_styling_errors"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        return sum(1 for line in flake8_output if any(code in line for code in [": E", ": F"]))


class StylingWarningPlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_styling_warnings"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        return sum(1 for line in flake8_output if ": W" in line)


class TotalIssuePlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_styling_issues"

    def extract(self, flake8_output: List[str], file_path: str) -> int:
        return len(flake8_output)


# ✅ Plugin registration order for stable metric vectors
ORDERED_PLUGINS: List[Flake8Plugin] = [
    UnusedVariablePlugin(),
    UnusedImportPlugin(),
    LongLinePlugin(),
    DocstringIssuePlugin(),
    NamingIssuePlugin(),
    StylingErrorPlugin(),
    StylingWarningPlugin(),
    TotalIssuePlugin(),
]


def load_plugins() -> List[Flake8Plugin]:
    """
    Loads all default Flake8 metric plugins.

    Returns:
        List[Flake8Plugin]: Instantiated plugins for metric extraction.
    """
    return ORDERED_PLUGINS


def get_flake8_metric_names() -> List[str]:
    """
    Returns:
        List[str]: Ordered names of Flake8 plugin metrics.
    """
    return [plugin.name() for plugin in ORDERED_PLUGINS]
