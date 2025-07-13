import ast
from .base import ASTMetricPlugin

class ChainedMethodCallPlugin(ASTMetricPlugin):
    """
    Counts chained method calls (e.g., obj.method1().method2()).
    """
    def name(self) -> str:
        return "chained_methods"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            for node in ast.walk(tree)
        )
