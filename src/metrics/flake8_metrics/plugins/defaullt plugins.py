"""
Default Flake8 metric plugins.
Each plugin defines a single feature to extract from raw Flake8 output.
"""

from metrics.flake8_metrics.plugins.base import Flake8Plugin


class UnusedVariablePlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_unused_variables"

    def extract(self, flake8_output, file_path):
        return sum(1 for line in flake8_output if ": F841 " in line)


class UnusedImportPlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_unused_imports"

    def extract(self, flake8_output, file_path):
        return sum(1 for line in flake8_output if ": F401 " in line)


class LongLinePlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_long_lines"

    def extract(self, flake8_output, file_path):
        return sum(1 for line in flake8_output if ": E501 " in line)


class DocstringIssuePlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_doc_string_issues"

    def extract(self, flake8_output, file_path):
        return sum(1 for line in flake8_output if ": D" in line)


class NamingIssuePlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_naming_issues"

    def extract(self, flake8_output, file_path):
        return sum(1 for line in flake8_output if ": N" in line)


class StylingErrorPlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_styling_errors"

    def extract(self, flake8_output, file_path):
        return sum(1 for line in flake8_output if any(code in line for code in [": E", ": F"]))


class StylingWarningPlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_styling_warnings"

    def extract(self, flake8_output, file_path):
        return sum(1 for line in flake8_output if ": W" in line)


class TotalIssuePlugin(Flake8Plugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_styling_issues"

    def extract(self, flake8_output, file_path):
        return len(flake8_output)


def load_plugins() -> list[Flake8Plugin]:
    """
    Loads all default Flake8 metric plugins.

    Returns:
        list[Flake8Plugin]: All plugin instances.
    """
    return [
        UnusedVariablePlugin(),
        UnusedImportPlugin(),
        LongLinePlugin(),
        DocstringIssuePlugin(),
        NamingIssuePlugin(),
        StylingErrorPlugin(),
        StylingWarningPlugin(),
        TotalIssuePlugin(),
    ]
