# File: code_analyser/src/metrics/ast_metrics/plugins/magic_method.py

import ast
from .base import ASTMetricPlugin  # ✅ required import


# 🧠 ML Signal: Magic methods (dunder methods) are structural indicators of class customisation and metaprogramming
# ⚠️ SAST Risk: Overuse or improper implementation of dunder methods can lead to unexpected behaviour and poor maintainability
# ✅ Best Practice: Plugin includes metadata and scoring to support GUI overlays and AI interpretation
class MagicMethodPlugin(ASTMetricPlugin):
    """
    Counts methods whose names start and end with double underscores (magic methods).

    Returns:
        int: Number of magic method definitions.
    """

    # ✅ Best Practice: Unique plugin name for registry and filtering
    plugin_name = "magic_methods"

    # ✅ Best Practice: Tags indicate usage in class design and metaprogramming analysis
    plugin_tags = ["magic", "dunder", "class_design", "special_methods"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Count all function definitions matching __name__ pattern
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("__")
            and node.name.endswith("__")
        )

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Thresholds based on magic method density
        count = self.visit(tree, code)
        if count == 0:
            return "low"
        elif count <= 3:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: More dunder methods → stronger signal → higher confidence
        count = self.visit(tree, code)
        return min(1.0, 0.2 * count)  # Maxes out at 5+
