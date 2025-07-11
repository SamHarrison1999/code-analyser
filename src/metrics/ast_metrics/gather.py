"""
Gather AST Metrics

Provides a helper function to extract and return a fixed-length, ordered
list of AST-based metric values for downstream use in ML models or CSV export.
"""

from typing import List
from metrics.ast_metrics.extractor import ASTMetricExtractor

# Fixed order of metric names for ML or CSV pipelines
# Must match feature expectations exactly.
_METRIC_ORDER: List[str] = [
    "functions",
    "classes",
    "function_docstrings",
    "class_docstrings",
    "module_docstring",
    "todo_comments",
    "assert_statements",
    "exceptions",
    "loops_conditionals",
    "nested_functions",
    "global_variables",
    "chained_methods",
    "lambda_functions",
    "magic_methods",
]

def gather_ast_metrics(file_path: str) -> List[int]:
    """
    Extract AST metrics from a Python file and return them as an ordered list.

    Args:
        file_path (str): Full path to the Python source file.

    Returns:
        list[int]: AST metrics in the fixed order defined by _METRIC_ORDER.
    """
    metrics = ASTMetricExtractor(file_path).extract()

    return [int(metrics.get(key, 0)) for key in _METRIC_ORDER]
