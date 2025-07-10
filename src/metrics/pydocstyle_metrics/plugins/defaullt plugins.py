"""
Default Pydocstyle metric plugins.
Each plugin extracts one metric from pydocstyle output.
"""

from metrics.pydocstyle_metrics.plugins.base import PydocstylePlugin


class MissingDocstringPlugin(PydocstylePlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_missing_doc_strings"

    def extract(self, pydocstyle_output, file_path):
        return sum(1 for line in pydocstyle_output if "Missing docstring" in line)


class TotalViolationsPlugin(PydocstylePlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_pydocstyle_violations"

    def extract(self, pydocstyle_output, file_path):
        return len(pydocstyle_output)


class CompliancePercentagePlugin(PydocstylePlugin):
    @classmethod
    def name(cls) -> str:
        return "percentage_of_compliance_with_docstring_style"

    def extract(self, pydocstyle_output, file_path):
        total = len(pydocstyle_output)
        missing = sum(1 for line in pydocstyle_output if "Missing docstring" in line)
        if total == 0:
            return 100.0
        return round(((total - missing) / total) * 100, 2)


def load_plugins() -> list[PydocstylePlugin]:
    """
    Loads all default Pydocstyle metric plugins.

    Returns:
        list[PydocstylePlugin]: All plugin instances.
    """
    return [
        MissingDocstringPlugin(),
        TotalViolationsPlugin(),
        CompliancePercentagePlugin(),
    ]
