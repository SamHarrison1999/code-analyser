# File: code_analyser/src/metrics/ast_metrics/plugins/assert_statements.py

import ast
from .base import ASTMetricPlugin  # ✅ required import


# 🧠 ML Signal: Plugins counting specific AST node types (e.g. assert) are strong signals for semantic code patterns
# ⚠️ SAST Risk: Overuse of 'assert' may indicate runtime logic testing in production code
# ✅ Best Practice: Classify plugin with descriptive metadata for discovery and filtering
class AssertStatementPlugin(ASTMetricPlugin):
    """
    Counts the number of assert statements in the AST.

    Returns:
        int: Number of `assert` statements.
    """

    # ✅ Best Practice: Plugin identifier for retrieval and display
    plugin_name = "assert_statements"

    # ✅ Best Practice: Plugin tags enable tag-based filtering (e.g., in GUI or CLI)
    plugin_tags = ["testing", "assert", "runtime_check"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Traverse AST and count Assert nodes explicitly
        return sum(isinstance(node, ast.Assert) for node in ast.walk(tree))

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Assign severity based on assert count thresholds
        count = self.visit(tree, code)
        if count == 0:
            return "low"
        elif count <= 5:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Provide a basic confidence function (capped at 1.0)
        count = self.visit(tree, code)
        return min(1.0, 0.1 * count)  # More asserts → more confidence, up to 1.0
