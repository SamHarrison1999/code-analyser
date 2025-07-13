import ast
import re
from .base import ASTMetricPlugin  # ✅ required import

class MagicMethodPlugin(ASTMetricPlugin):
    """
    Counts methods whose names start and end with double underscores (magic methods).

    Returns:
        int: Number of magic method definitions.
    """
    def name(self) -> str:
        return "magic_methods"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("__")
            and node.name.endswith("__")
        )