# File: src/metrics/ast_metrics/plugins/default_plugins.py

import ast
import re
from .base import ASTMetricPlugin


class ModuleDocstringPlugin(ASTMetricPlugin):
    """
    Checks whether the module has a top-level docstring.

    Returns 1 if a module docstring exists, otherwise 0.
    """

    def name(self) -> str:
        return "module_docstring"

    def visit(self, tree: ast.AST, code: str) -> int:
        return int(bool(ast.get_docstring(tree)))


class TodoCommentPlugin(ASTMetricPlugin):
    """
    Counts the number of TODO or FIXME comments in the source code.

    Searches for single-line comments that contain either term (case-insensitive).
    """

    def name(self) -> str:
        return "todo_comments"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(
            1 for line in code.splitlines()
            if line.strip().startswith("#") and re.search(r'\b(TODO|FIXME)\b', line, re.IGNORECASE)
        )


class NestedFunctionPlugin(ASTMetricPlugin):
    """
    Counts the number of nested functions (i.e., functions defined within other functions).
    """

    def name(self) -> str:
        return "nested_functions"

    def visit(self, tree: ast.AST, code: str) -> int:
        def count_nested(node: ast.AST, parent_is_function: bool = False) -> int:
            count = 0
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if parent_is_function:
                        count += 1
                    count += count_nested(child, True)
                else:
                    count += count_nested(child, parent_is_function)
            return count

        return count_nested(tree)
