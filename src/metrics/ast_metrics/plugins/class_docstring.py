import ast
import re
from .base import ASTMetricPlugin  # ✅ required import

class ClassDocstringPlugin(ASTMetricPlugin):
    """
    Counts the number of classes that have a docstring.

    Returns:
        int: Number of classes with a docstring.
    """
    def name(self) -> str:
        return "class_docstrings"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and ast.get_docstring(node) is not None
        )