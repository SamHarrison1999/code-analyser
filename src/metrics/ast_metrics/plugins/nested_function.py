# File: code_analyser/src/metrics/ast_metrics/plugins/nested_function.py

import ast
from .base import ASTMetricPlugin  # ✅ required import


# 🧠 ML Signal: Nested function definitions are key indicators of functional encapsulation and closure use
# ⚠️ SAST Risk: Overuse of nested functions may reduce readability and testability, and hide scope side-effects
# ✅ Best Practice: Use structured metadata and confidence/severity scoring for overlays
class NestedFunctionPlugin(ASTMetricPlugin):
    """
    Counts the number of functions defined inside other functions.

    Returns:
        int: Number of nested function definitions.
    """

    # ✅ Best Practice: Unique plugin name for plugin discovery and export
    plugin_name = "nested_functions"

    # ✅ Best Practice: Categorisation tags for GUI filtering and AI overlays
    plugin_tags = ["structure", "functional", "scope", "complexity"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Recursively count nested (inner) functions
        def count_nested(node: ast.AST, inside_function: bool = False) -> int:
            count = 0
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if inside_function:
                        count += 1
                    count += count_nested(child, True)
                else:
                    count += count_nested(child, inside_function)
            return count

        return count_nested(tree)

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Threshold severity by number of nested function blocks
        count = self.visit(tree, code)
        if count == 0:
            return "low"
        elif count <= 3:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Higher count increases confidence up to a max of 1.0
        count = self.visit(tree, code)
        return min(1.0, 0.25 * count)  # Full confidence after 4+ nested defs
