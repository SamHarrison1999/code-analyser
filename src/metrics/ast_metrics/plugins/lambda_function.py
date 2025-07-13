import ast
import re
from .base import ASTMetricPlugin  # ✅ required import

class LambdaFunctionPlugin(ASTMetricPlugin):
    """
    Counts the number of lambda expressions in the AST.

    Returns:
        int: Number of lambda expressions.
    """
    def name(self) -> str:
        return "lambda_functions"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(isinstance(node, ast.Lambda) for node in ast.walk(tree))
