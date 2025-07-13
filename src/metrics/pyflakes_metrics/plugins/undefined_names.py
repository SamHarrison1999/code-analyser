from metrics.pyflakes_metrics.plugins.base import PyflakesMetricPlugin

class UndefinedNamesMetricPlugin(PyflakesMetricPlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_undefined_names"

    def extract(self, pyflakes_output: list[str], file_path: str) -> int:
        return sum(1 for line in pyflakes_output if "undefined name" in line.lower())
