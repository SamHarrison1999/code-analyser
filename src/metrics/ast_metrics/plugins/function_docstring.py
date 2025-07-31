# File: code_analyser/src/metrics/ast_metrics/plugins/function_docstring.py

import ast
from .base import ASTMetricPlugin


# 🧠 ML Signal: Function-level docstrings are strong predictors of documentation quality and maintainability
# ⚠️ SAST Risk: Missing docstrings reduce testability, code audit readiness, and self-documentation
# ✅ Best Practice: Include plugin metadata and scoring methods for full overlay support
class FunctionDocstringPlugin(ASTMetricPlugin):
    """
    Counts the number of functions (including async) that have docstrings.
    """

    # ✅ Best Practice: Unique plugin identifier for dynamic discovery
    plugin_name = "function_docstrings"

    # ✅ Best Practice: Tags allow categorisation and filtering by code quality dimension
    plugin_tags = ["documentation", "functions", "quality", "maintainability"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Count sync and async functions with non-empty docstrings
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and bool(ast.get_docstring(node))
        )

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Classify severity by docstring coverage ratio
        total = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(tree)
        )
        documented = self.visit(tree, code)
        if total == 0:
            return "low"
        ratio = documented / total
        if ratio >= 0.8:
            return "low"
        elif ratio >= 0.5:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Confidence reflects how well-covered the functions are
        total = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(tree)
        )
        documented = self.visit(tree, code)
        if total == 0:
            return 1.0  # If no functions, result is trivially confident
        return round(documented / total, 2)
