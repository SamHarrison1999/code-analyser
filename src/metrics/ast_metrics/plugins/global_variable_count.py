# File: code_analyser/src/metrics/ast_metrics/plugins/global_variable_count.py

import ast
from .base import ASTMetricPlugin


# 🧠 ML Signal: Use of global variables is often linked to poor encapsulation and testability issues
# ⚠️ SAST Risk: Global state can introduce hidden dependencies and race conditions in concurrent environments
# ✅ Best Practice: Register metadata for overlay filtering and scoring
class GlobalVariablePlugin(ASTMetricPlugin):
    """
    Counts the number of global variable declarations.
    """

    # ✅ Best Practice: Plugin identifier for dynamic discovery
    plugin_name = "global_variables"

    # ✅ Best Practice: Tags for classification in GUI, CLI, and overlays
    plugin_tags = ["state", "global", "encapsulation", "complexity"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Count declared global variable names in all global statements
        return sum(len(node.names) for node in ast.walk(tree) if isinstance(node, ast.Global))

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Thresholds classify risk from global variable use
        count = self.visit(tree, code)
        if count == 0:
            return "low"
        elif count <= 2:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Confidence based on the number of distinct globals
        count = self.visit(tree, code)
        return min(1.0, 0.25 * count)  # Full confidence at 4+
