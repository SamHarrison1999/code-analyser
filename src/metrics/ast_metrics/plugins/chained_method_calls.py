# File: code_analyser/src/metrics/ast_metrics/plugins/chained_method_calls.py

import ast
from .base import ASTMetricPlugin


# 🧠 ML Signal: Method chaining patterns often indicate fluent interfaces or compact logic
# ⚠️ SAST Risk: Deep chains can reduce readability and obscure side effects
# ✅ Best Practice: Plugin should declare metadata and scoring logic for filtering and AI overlays
class ChainedMethodCallPlugin(ASTMetricPlugin):
    """
    Counts chained method calls (e.g., obj.method1().method2()).
    """

    # ✅ Best Practice: Define unique plugin name for registry use
    plugin_name = "chained_methods"

    # ✅ Best Practice: Assign tags for filtering by metric categories
    plugin_tags = ["readability", "fluent_interface", "method_chaining"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Count method chains with two or more attribute calls
        return sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            for node in ast.walk(tree)
        )

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Assign severity based on number of chains
        count = self.visit(tree, code)
        if count == 0:
            return "low"
        elif count <= 3:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Confidence increases with more detected method chains
        count = self.visit(tree, code)
        return min(1.0, 0.15 * count)  # Capped at 1.0
