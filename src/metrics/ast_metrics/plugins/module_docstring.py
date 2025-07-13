import ast
import re
from .base import ASTMetricPlugin  # ✅ required import

class ModuleDocstringPlugin(ASTMetricPlugin):
    """
    Checks whether the module has a top-level docstring.

    Returns:
        int: 1 if a module docstring exists, otherwise 0.
    """
    def name(self) -> str:
        return "module_docstring"

    def visit(self, tree: ast.AST, code: str) -> int:
        return int(ast.get_docstring(tree) is not None)