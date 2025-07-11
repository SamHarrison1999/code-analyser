from .base import PyflakesPlugin


class UndefinedNamesPlugin(PyflakesPlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_undefined_names"

    def extract(self, pyflakes_output: list[str], file_path: str) -> int:
        return sum("undefined name" in line.lower() for line in pyflakes_output)


class SyntaxErrorsPlugin(PyflakesPlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_syntax_errors"

    def extract(self, pyflakes_output: list[str], file_path: str) -> int:
        return sum("syntax error" in line.lower() for line in pyflakes_output)


def load_plugins() -> list[PyflakesPlugin]:
    """
    Instantiate and return all default Pyflakes metric plugins.
    """
    return [
        UndefinedNamesPlugin(),
        SyntaxErrorsPlugin(),
    ]
