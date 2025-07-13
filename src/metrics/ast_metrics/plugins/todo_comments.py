import ast
import re
from .base import ASTMetricPlugin  # ✅ required import

class TodoCommentPlugin(ASTMetricPlugin):
    """
    Counts the number of TODO or FIXME comments in the source code.

    Only lines that begin with '#' are considered, case-insensitively.
    """
    def name(self) -> str:
        return "todo_comments"

    def visit(self, tree: ast.AST, code: str) -> int:
        return sum(
            1
            for line in code.splitlines()
            if line.strip().startswith("#") and re.search(r"\b(?:TODO|FIXME)\b", line, re.IGNORECASE)
        )