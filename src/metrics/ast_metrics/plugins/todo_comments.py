# File: code_analyser/src/metrics/ast_metrics/plugins/todo_comments.py

import ast
import re
from .base import ASTMetricPlugin  # ✅ required import


# 🧠 ML Signal: TODO/FIXME comments are predictive of incomplete or transitional code states
# ⚠️ SAST Risk: Leftover TODOs or FIXMEs can indicate unresolved logic, vulnerabilities, or known defects
# ✅ Best Practice: Include metadata and scoring to enable AI overlays and dashboard filtering
class TodoCommentPlugin(ASTMetricPlugin):
    """
    Counts the number of TODO or FIXME comments in the source code.

    Only lines that begin with '#' are considered, case-insensitively.
    """

    # ✅ Best Practice: Unique identifier for discovery and registration
    plugin_name = "todo_comments"

    # ✅ Best Practice: Tags useful for quality and tracking of unresolved code
    plugin_tags = ["comments", "incomplete", "quality", "reminder", "sast"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Scan for comment lines with TODO/FIXME using regex
        return sum(
            1
            for line in code.splitlines()
            if line.strip().startswith("#")
            and re.search(r"\b(?:TODO|FIXME)\b", line, re.IGNORECASE)
        )

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: High severity if many unresolved TODOs are found
        count = self.visit(tree, code)
        if count == 0:
            return "low"
        elif count <= 3:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Simple linear scaling based on number of matches
        count = self.visit(tree, code)
        return min(1.0, 0.25 * count)  # Max confidence at 4+
