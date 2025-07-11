import ast
import re
from .base import ASTMetricPlugin


class ModuleDocstringPlugin(ASTMetricPlugin):
    """
    Checks whether the module has a top-level docstring.

    Returns:
        1 if a module docstring exists, otherwise 0.
    """
    def name(self) -> str:
        return "module_docstring"

    def visit(self, tree: ast.AST, code: str) -> int:
        return int(ast.get_docstring(tree) is not None)


class TodoCommentPlugin(ASTMetricPlugin):
    """
    Counts the number of TODO or FIXME comments in the source code.

    Only lines that begin with '#' are considered, case-insensitively.
    """
    def name(self) -> str:
        return "todo_comments"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(
            1
            for line in code.splitlines()
            if line.strip().startswith("#") and re.search(r"\b(?:TODO|FIXME)\b", line, re.IGNORECASE)
        )


class NestedFunctionPlugin(ASTMetricPlugin):
    """
    Counts the number of functions defined inside other functions.
    """
    def name(self) -> str:
        return "nested_functions"

    def visit(self, tree: ast.AST, code: str) -> int:
        def count_nested(node: ast.AST, inside_function: bool = False) -> int:
            """
            Recursively count nested functions.

            Args:
                node (ast.AST): Current AST node.
                inside_function (bool): Whether the parent is a function node.

            Returns:
                int: Count of nested functions under this node.
            """
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


class LambdaFunctionPlugin(ASTMetricPlugin):
    """
    Counts the number of lambda expressions in the AST.
    """
    def name(self) -> str:
        return "lambda_functions"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(isinstance(node, ast.Lambda) for node in ast.walk(tree))


class MagicMethodPlugin(ASTMetricPlugin):
    """
    Counts methods whose names start and end with double underscores (magic methods).
    """
    def name(self) -> str:
        return "magic_methods"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("__") and node.name.endswith("__")
        )


class AssertStatementPlugin(ASTMetricPlugin):
    """
    Counts the number of assert statements in the AST.
    """
    def name(self) -> str:
        return "assert_statements"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(isinstance(node, ast.Assert) for node in ast.walk(tree))


class ClassDocstringPlugin(ASTMetricPlugin):
    """
    Counts the number of classes that have a docstring.
    """
    def name(self) -> str:
        return "class_docstrings"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and ast.get_docstring(node) is not None
        )
