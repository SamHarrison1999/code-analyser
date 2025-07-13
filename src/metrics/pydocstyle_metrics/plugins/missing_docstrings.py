from typing import List
from .base import PydocstyleMetricPlugin


class MissingDocstringMetricPlugin(PydocstyleMetricPlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_missing_doc_strings"

    def extract(self, pydocstyle_output: List[str], file_path: str) -> int:
        return sum(1 for line in pydocstyle_output if "Missing docstring" in line)
