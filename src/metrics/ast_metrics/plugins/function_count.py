# File: code_analyser/src/metrics/ast_metrics/plugins/function_count.py

import ast
from .base import ASTMetricPlugin


# 🧠 ML Signal: Function count reveals structural complexity and modularity
# ⚠️ SAST Risk: Large numbers of functions may indicate bloated files or insufficient cohesion
# ✅ Best Practice: Define metadata and scoring for GUI overlays, filters, and summaries
class FunctionCountPlugin(ASTMetricPlugin):
    """
    Counts the number of top-level and nested function definitions.
    """

    # ✅ Best Practice: Plugin identifier used in loaders and filtering
    plugin_name = "functions"

    # ✅ Best Practice: Tags for filtering, overlays, and CLI grouping
    plugin_tags = ["structure", "modularity", "functions", "complexity"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Count both sync and async function definitions
        return sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)
        )

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Threshold severity based on function count
        count = self.visit(tree, code)
        if count == 0:
            return "low"
        elif count <= 10:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Confidence increases with more observed functions
        count = self.visit(tree, code)
        return min(1.0, 0.1 * count)  # Reaches full confidence at 10+
