import ast
from .base import ASTMetricPlugin

class GlobalVariablePlugin(ASTMetricPlugin):
    """
    Counts the number of global variable declarations.
    """
    def name(self) -> str:
        return "global_variables"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(len(node.names) for node in ast.walk(tree) if isinstance(node, ast.Global))
