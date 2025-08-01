# File: code_analyser/src/metrics/ast_metrics/plugins/module_docstring.py

import ast
from .base import ASTMetricPlugin  # ✅ required import


# 🧠 ML Signal: Module-level docstrings are key indicators of maintainability and intent documentation
# ⚠️ SAST Risk: Missing module docstrings reduce clarity and weaken automated analysis for codebases
# ✅ Best Practice: Include plugin metadata and dynamic scoring logic for overlay integration
class ModuleDocstringPlugin(ASTMetricPlugin):
    """
    Checks whether the module has a top-level docstring.

    Returns:
        int: 1 if a module docstring exists, otherwise 0.
    """

    # ✅ Best Practice: Unique identifier for plugin discovery
    plugin_name = "module_docstring"

    # ✅ Best Practice: Tags highlight documentation and maintainability relevance
    plugin_tags = ["documentation", "module", "quality", "maintainability"]

    def name(self) -> str:
        return self.plugin_name

    def visit(self, tree: ast.AST, code: str) -> int:
        # ✅ Best Practice: Use AST docstring retrieval at module level
        return int(ast.get_docstring(tree) is not None)

    def severity_level(self, tree: ast.AST, code: str) -> str:
        # ✅ Best Practice: Simple severity flag based on absence/presence
        has_doc = self.visit(tree, code)
        return "low" if has_doc else "high"

    def confidence_score(self, tree: ast.AST, code: str) -> float:
        # ✅ Best Practice: Binary confidence since signal is explicit
        return 1.0
