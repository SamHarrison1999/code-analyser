import ast
from .base import ASTMetricPlugin

class ExceptionCountPlugin(ASTMetricPlugin):
    """
    Counts the number of exception handlers (except blocks).
    """
    def name(self) -> str:
        return "exceptions"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(isinstance(node, ast.ExceptHandler) for node in ast.walk(tree))
