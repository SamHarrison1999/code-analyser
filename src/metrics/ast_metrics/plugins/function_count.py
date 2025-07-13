import ast
from .base import ASTMetricPlugin

class FunctionCountPlugin(ASTMetricPlugin):
    """
    Counts the number of top-level and nested function definitions.
    """
    def name(self) -> str:
        return "functions"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
