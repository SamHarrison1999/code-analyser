# File: code_analyser/src/metrics/ast_metrics/plugins/lambda_function.py

import ast
from .base import ASTMetricPlugin  # ✅ required import


# 🧠 ML Signal: Lambda usage may indicate functional programming idioms or overly concise logic
# ⚠️ SAST Risk: Excessive use of lambdas can hinder readability and traceability during debugging
# ✅ Best Practice: Metadata supports plugin discovery and overlay filtering
class LambdaFunctionPlugin(ASTMetricPlugin):
    """
    Counts the number of lambda expressions in the AST.

    Returns:
        int: Number of lambda expressions.
    """

    # ✅ Best Practice: Plugin identifier for loader compatibility
    plugin_name = "lambda_functions"

    # ✅ Best Practice: Tags describe category and concerns (functional logic, compactness)
    plugin_tags = ["functional", "compactness", "lambda", "readability"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Count Lambda nodes directly via AST walk
        return sum(isinstance(node, ast.Lambda) for node in ast.walk(tree))

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Threshold-based severity based on lambda usage
        count = self.visit(tree, code)
        if count == 0:
            return "low"
        elif count <= 3:
            return "medium"
        else:
            return "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Higher lambda count → higher confidence
        count = self.visit(tree, code)
        return min(1.0, 0.2 * count)  # Max confidence at 5+
