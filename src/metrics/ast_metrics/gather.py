"""
Gather AST Metrics

Provides a helper function to extract and return a fixed-length, ordered
list of AST-based metric values for downstream use in ML models or CSV export.
"""

from typing import List
from metrics.ast_metrics.extractor import ASTMetricExtractor

# ✅ Best Practice: Maintain strict metric order for ML and CSV alignment
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
    Extract AST metrics from a Python file and return them in fixed order.

    Args:
        file_path (str): Absolute or relative path to the Python source file.

    Returns:
        List[int]: Ordered list of AST metric values aligned with _METRIC_ORDER.
    """
    extractor = ASTMetricExtractor(file_path)
    metrics = extractor.extract()
    return [int(metrics.get(key, 0)) for key in _METRIC_ORDER]

def get_ast_metric_names() -> List[str]:
    """
    Return the list of metric names used in gather_ast_metrics, in strict order.

    Returns:
        List[str]: Ordered metric name list for consistent model or CSV output.
    """
    return _METRIC_ORDER
