import ast
import re
from .base import ASTMetricPlugin  # ✅ required import

class NestedFunctionPlugin(ASTMetricPlugin):
    """
    Counts the number of functions defined inside other functions.

    Returns:
        int: Number of nested function definitions.
    """
    def name(self) -> str:
        return "nested_functions"

    def visit(self, tree: ast.AST, code: str) -> int:
        def count_nested(node: ast.AST, inside_function: bool = False) -> int:
            count = 0
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if inside_function:
                        count += 1
                    count += count_nested(child, True)
                else:
                    count += count_nested(child, inside_function)
            return count

        return count_nested(tree)