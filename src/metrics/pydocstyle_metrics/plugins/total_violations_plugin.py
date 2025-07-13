from typing import List
from .base import PydocstyleMetricPlugin


class TotalViolationsMetricPlugin(PydocstyleMetricPlugin):
    @classmethod
    def name(cls) -> str:
        return "number_of_pydocstyle_violations"

    def extract(self, pydocstyle_output: List[str], file_path: str) -> int:
        return len(pydocstyle_output)
