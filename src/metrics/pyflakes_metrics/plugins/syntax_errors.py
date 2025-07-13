from metrics.pyflakes_metrics.plugins.base import PyflakesMetricPlugin

class SyntaxErrorsMetricPlugin(PyflakesMetricPlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_syntax_errors"

    def extract(self, pyflakes_output: list[str], file_path: str) -> int:
        return sum(1 for line in pyflakes_output if "syntax error" in line.lower())
