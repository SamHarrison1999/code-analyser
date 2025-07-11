"""
Default Pyflakes metric plugins.
Each plugin extracts one metric from pyflakes output.
"""

from metrics.pyflakes_metrics.plugins.base import PyflakesPlugin


class UndefinedNamesPlugin(PyflakesPlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_undefined_names"

    def extract(self, pyflakes_output: list[str], file_path: str) -> int:
        """
        Count the number of undefined name warnings.

        Args:
            pyflakes_output (list[str]): Output lines from pyflakes.
            file_path (str): Path to the analysed file.

        Returns:
            int: Number of 'undefined name' occurrences.
        """
        return sum("undefined name" in line.lower() for line in pyflakes_output)


class SyntaxErrorsPlugin(PyflakesPlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_syntax_errors"

    def extract(self, pyflakes_output: list[str], file_path: str) -> int:
        """
        Count the number of syntax error messages.

        Args:
            pyflakes_output (list[str]): Output lines from pyflakes.
            file_path (str): Path to the analysed file.

        Returns:
            int: Number of syntax errors found.
        """
        return sum("syntax error" in line.lower() for line in pyflakes_output)


def load_plugins() -> list[PyflakesPlugin]:
    """
    Loads all default Pyflakes metric plugins.

    Returns:
        list[PyflakesPlugin]: All plugin instances.
    """
    return [
        UndefinedNamesPlugin(),
        SyntaxErrorsPlugin(),
    ]


# ✅ Exported list of plugin instances for unified access
plugins = load_plugins()
