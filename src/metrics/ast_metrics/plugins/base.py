# File: src/metrics/ast_metrics/plugins/base.py

import ast
from abc import ABC, abstractmethod


class ASTMetricPlugin(ABC):
    """
    Abstract base class for AST metric plugins.

    Each plugin must define:
    - a unique name for the metric (used as dictionary key)
    - a `visit()` method that computes the metric from the AST
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: Unique metric name this plugin computes.
        """
        raise NotImplementedError("Plugin must define a metric name.")

    @abstractmethod
    def visit(self, tree: ast.AST, code: str) -> int:
        """
        Computes the metric value by inspecting the AST and/or source code.

        Args:
            tree (ast.AST): The parsed abstract syntax tree.
            code (str): The original source code string.

        Returns:
            int: The computed value for this metric.
        """
        raise NotImplementedError("Plugin must implement visit().")
