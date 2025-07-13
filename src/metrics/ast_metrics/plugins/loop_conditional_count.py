import ast
from .base import ASTMetricPlugin

class LoopConditionalPlugin(ASTMetricPlugin):
    """
    Counts the number of loops (for, while) and conditionals (if).
    """
    def name(self) -> str:
        return "loops_conditionals"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(isinstance(node, (ast.For, ast.While, ast.If)) for node in ast.walk(tree))
