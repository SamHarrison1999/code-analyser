# File: code_analyser/src/metrics/ast_metrics/plugins/class_count.py

import ast
from .base import ASTMetricPlugin


# 🧠 ML Signal: Class definitions provide high-level semantic structure signals for supervised learning
# ⚠️ SAST Risk: Excessive class definitions may indicate unnecessary complexity or overengineering
# ✅ Best Practice: Include plugin metadata for filtering and dynamic loading
class ClassCountPlugin(ASTMetricPlugin):
    """
    Counts the number of class definitions.
    """

    # ✅ Best Practice: Unique plugin identifier
    plugin_name = "classes"

    # ✅ Best Practice: Tags help categorise this metric for readability, OO structure, etc.
    plugin_tags = ["structure", "object_oriented", "complexity"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Use AST traversal to count class definitions
        return sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Determine severity based on thresholds
        count = self.visit(tree, code)
        if count == 0:
            return "low"
        elif count <= 5:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Scaled confidence based on number of detected classes
        count = self.visit(tree, code)
        return min(1.0, 0.2 * count)  # Caps at 1.0 for 5+ classes
