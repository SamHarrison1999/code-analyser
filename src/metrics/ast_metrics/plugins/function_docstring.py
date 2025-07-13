import ast
from .base import ASTMetricPlugin

class FunctionDocstringPlugin(ASTMetricPlugin):
    """
    Counts the number of functions (including async) that have docstrings.
    """
    def name(self) -> str:
        return "function_docstrings"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and bool(ast.get_docstring(node))
        )
