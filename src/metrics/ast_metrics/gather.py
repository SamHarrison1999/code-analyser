# File: src/metrics/ast_metrics/gather.py

from typing import List
from metrics.ast_metrics.extractor import ASTMetricExtractor

# Fixed order of metric names for ML or CSV pipelines
_METRIC_ORDER = [
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
    Extracts AST metrics from a Python file and returns them
    as a list in a fixed order, ready for machine learning or export.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[int]: Ordered list of AST metric values.
    """
    metrics = ASTMetricExtractor(file_path).extract()

    # Return metrics in a consistent order with safe fallback
    return [metrics.get(key, 0) for key in _METRIC_ORDER]
