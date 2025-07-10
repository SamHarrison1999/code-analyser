# File: metrics/pydocstyle_metrics/plugins/__init__.py

from metrics.pydocstyle_metrics.gather import gather_pydocstyle_metrics

METRIC_NAME_LIST = [
    "number_of_pydocstyle_violations",
    "number_of_missing_doc_strings",
    "percentage_of_compliance_with_docstring_style"
]

def get_pydocstyle_metric_names() -> list[str]:
    return METRIC_NAME_LIST

def pydocstyle_metric_plugin(file_path: str) -> list[float]:
    return gather_pydocstyle_metrics(file_path)
