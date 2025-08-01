# File: code_analyser/src/metrics/ast_metrics/plugins/base.py

import ast
from abc import ABC, abstractmethod


# 🧠 ML Signal: Base class structure defines reusable contract for supervised model features
# ✅ Best Practice: Use an abstract base class to define extensible plugin interfaces with optional AI support
class ASTMetricPlugin(ABC):
    """
    Abstract base class for AST metric plugins.

    Each plugin must:
    - Provide a unique metric name via the `name()` method.
    - Implement `visit()` to compute the metric using an AST and source code.
    - Optionally define `plugin_name`, `plugin_tags`, `confidence_score`, and `severity_level`.
    """

    # ✅ Best Practice: Optional plugin metadata
    plugin_name: str = ""
    plugin_tags: list[str] = []

    # 🧠 ML Signal: Confidence and severity support enables plugin-level AI overlays
    # These can be overridden in subclasses or computed dynamically
    def confidence_score(self, tree: ast.AST, code: str) -> float:
        """
        Return a float confidence score [0.0, 1.0] for the metric.

        Override in plugins that support confidence estimation.

        Returns:
            float: Confidence in metric value (default: 1.0)
        """
        return 1.0

    def severity_level(self, tree: ast.AST, code: str) -> str:
        """
        Return a string representing severity of findings.

        Valid levels: 'low', 'medium', 'high'.

        Returns:
            str: Severity level (default: 'low')
        """
        return "low"

    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of the metric provided by this plugin.

        Returns:
            str: The name of the metric (used as a dictionary key).
        """
        raise NotImplementedError("Subclasses must implement the name() method.")

    @abstractmethod
    def visit(self, tree: ast.AST, code: str) -> int:
        """
        Compute the metric value by traversing the AST or analysing the source code.

        Args:
            tree (ast.AST): Parsed abstract syntax tree of the source file.
            code (str): Raw source code as a string.

        Returns:
            int: Computed integer value for the metric.
        """
        raise NotImplementedError("Subclasses must implement the visit() method.")
