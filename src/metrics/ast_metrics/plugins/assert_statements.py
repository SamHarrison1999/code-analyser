import ast
import re
from .base import ASTMetricPlugin  # ✅ required import

class AssertStatementPlugin(ASTMetricPlugin):
    """
    Counts the number of assert statements in the AST.

    Returns:
        int: Number of `assert` statements.
    """
    def name(self) -> str:
        return "assert_statements"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
