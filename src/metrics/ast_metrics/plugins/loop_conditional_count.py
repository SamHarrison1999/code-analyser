# File: code_analyser/src/metrics/ast_metrics/plugins/loop_conditional_count.py

import ast
from .base import ASTMetricPlugin


# 🧠 ML Signal: Loop and conditional density strongly correlates with control flow complexity
# ⚠️ SAST Risk: Overuse of branching and loops may introduce cognitive complexity or unreachable paths
# ✅ Best Practice: Define metadata for plugin discovery, filtering, and overlay support
class LoopConditionalPlugin(ASTMetricPlugin):
    """
    Counts the number of loops (for, while) and conditionals (if).
    """

    # ✅ Best Practice: Unique plugin identifier
    plugin_name = "loops_conditionals"

    # ✅ Best Practice: Tags categorise this as a control-flow complexity metric
    plugin_tags = ["control_flow", "branching", "complexity", "structure"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Count loop and conditional statements using AST
        return sum(
            isinstance(node, (ast.For, ast.While, ast.If)) for node in ast.walk(tree)
        )

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Classify severity based on total loop/conditional count
        count = self.visit(tree, code)
        if count == 0:
            return "low"
        elif count <= 10:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Confidence increases with count, capped at 1.0
        count = self.visit(tree, code)
        return min(1.0, 0.1 * count)  # Maxes out at 10+
