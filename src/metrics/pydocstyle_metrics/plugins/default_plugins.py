"""
Default Pydocstyle metric plugins.
Each plugin extracts one metric from pydocstyle output.
"""

from metrics.pydocstyle_metrics.plugins.base import PydocstylePlugin


class MissingDocstringPlugin(PydocstylePlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_missing_doc_strings"

    def extract(self, pydocstyle_output: list[str], file_path: str) -> int:
        """
        Count the number of missing docstring violations.

        Args:
            pydocstyle_output (list[str]): Lines of output from Pydocstyle.
            file_path (str): Path to the analysed source file.

        Returns:
            int: Count of violations containing "Missing docstring".
        """
        return sum(1 for line in pydocstyle_output if "Missing docstring" in line)


class TotalViolationsPlugin(PydocstylePlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_pydocstyle_violations"

    def extract(self, pydocstyle_output: list[str], file_path: str) -> int:
        """
        Count the total number of docstring violations.

        Args:
            pydocstyle_output (list[str]): Lines of output from Pydocstyle.
            file_path (str): Path to the analysed source file.

        Returns:
            int: Total number of violations reported.
        """
        return len(pydocstyle_output)


class CompliancePercentagePlugin(PydocstylePlugin):
    @classmethod
    def name(cls) -> str:
        return "percentage_of_compliance_with_docstring_style"

    def extract(self, pydocstyle_output: list[str], file_path: str) -> float:
        """
        Calculate the percentage of Pydocstyle compliance.

        Args:
            pydocstyle_output (list[str]): Lines of output from Pydocstyle.
            file_path (str): Path to the analysed source file.

        Returns:
            float: Percentage of violations that are not missing docstrings.
        """
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


__all__ = [
    "MissingDocstringPlugin",
    "TotalViolationsPlugin",
    "CompliancePercentagePlugin",
    "load_plugins",
]
