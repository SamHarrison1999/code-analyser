# File: src/metrics/ast_metrics/gather.py

from typing import List
from metrics.ast_metrics.extractor import ASTMetricExtractor


def gather_ast_metrics(file_path: str) -> List[int]:
    """
    Extracts AST metrics from a Python file and returns them
    as a list in a fixed order, ready for machine learning consumption.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        List[int]: Ordered list of AST metric values.
    """
    metrics = ASTMetricExtractor(file_path).extract()

    # Return metrics in a consistent order
    return [
        metrics["functions"],
        metrics["classes"],
        metrics["function_docstrings"],
        metrics["class_docstrings"],
        metrics["module_docstring"],
        metrics["todo_comments"],
        metrics["assert_statements"],
        metrics["exceptions"],
        metrics["loops_conditionals"],
        metrics["nested_functions"],
        metrics["global_variables"],
        metrics["chained_methods"],
        metrics["lambda_functions"],
        metrics["magic_methods"],
    ]
