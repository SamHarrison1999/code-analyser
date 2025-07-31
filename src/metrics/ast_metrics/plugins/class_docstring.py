# File: code_analyser/src/metrics/ast_metrics/plugins/class_docstring.py

import ast
from .base import ASTMetricPlugin  # ✅ required import


# 🧠 ML Signal: Presence of docstrings is a strong signal for code maintainability and documentation quality
# ⚠️ SAST Risk: Missing class docstrings can reduce auditability and increase onboarding difficulty
# ✅ Best Practice: Plugin metadata and scoring support AI overlays and GUI filtering
class ClassDocstringPlugin(ASTMetricPlugin):
    """
    Counts the number of classes that have a docstring.

    Returns:
        int: Number of classes with a docstring.
    """

    # ✅ Best Practice: Plugin identifier for registry use
    plugin_name = "class_docstrings"

    # ✅ Best Practice: Tags for filtering in dashboards and AI overlays
    plugin_tags = ["documentation", "class", "quality", "maintainability"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Count class definitions that have a docstring
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and ast.get_docstring(node) is not None
        )

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Determine severity based on missing docstring percentage
        total_classes = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        documented = self.visit(tree, code)
        if total_classes == 0:
            return "low"
        ratio = documented / total_classes
        if ratio >= 0.8:
            return "low"
        elif ratio >= 0.5:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Confidence increases with number of documented classes
        total_classes = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        documented = self.visit(tree, code)
        if total_classes == 0:
            return 1.0  # If no classes, confidently report zero
        return round(documented / total_classes, 2)
