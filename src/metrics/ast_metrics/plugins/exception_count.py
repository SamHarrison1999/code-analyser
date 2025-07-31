# File: code_analyser/src/metrics/ast_metrics/plugins/exception_count.py

import ast
from .base import ASTMetricPlugin


# 🧠 ML Signal: Exception handling frequency can signal error-prone logic or defensive coding patterns
# ⚠️ SAST Risk: Numerous try/except blocks may conceal code that frequently fails or lacks validation
# ✅ Best Practice: Use structured plugin metadata and scoring for extensibility and AI overlays
class ExceptionCountPlugin(ASTMetricPlugin):
    """
    Counts the number of exception handlers (except blocks).
    """

    # ✅ Best Practice: Unique plugin name for discovery and filtering
    plugin_name = "exceptions"

    # ✅ Best Practice: Tags useful for tooling and visualisation
    plugin_tags = ["error_handling", "exceptions", "robustness", "complexity"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Count all AST nodes that represent exception handlers
        return sum(isinstance(node, ast.ExceptHandler) for node in ast.walk(tree))

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Classify based on exception handler density
        count = self.visit(tree, code)
        if count == 0:
            return "low"
        elif count <= 3:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Increase confidence with more observed try/except patterns
        count = self.visit(tree, code)
        return min(1.0, 0.25 * count)  # Caps at 1.0
