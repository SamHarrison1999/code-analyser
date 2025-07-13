import ast
from .base import ASTMetricPlugin

class ClassCountPlugin(ASTMetricPlugin):
    """
    Counts the number of class definitions.
    """
    def name(self) -> str:
        return "classes"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
