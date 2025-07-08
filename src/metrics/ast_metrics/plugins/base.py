# File: src/metrics/ast_metrics/plugins/base.py

import ast
from abc import ABC, abstractmethod


class ASTMetricPlugin(ABC):
    """
    Abstract base class for AST metric plugins.
    Each plugin must define a unique metric name and implement a `visit` method.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: The name of the metric this plugin computes.
        """
        pass

    @abstractmethod
    def visit(self, tree: ast.AST, code: str) -> int:
        """
        Computes the metric value by inspecting the AST and/or source code.

        Args:
            tree (ast.AST): The parsed abstract syntax tree.
            code (str): The raw source code as a string.

        Returns:
            int: The computed metric value.
        """
        pass
