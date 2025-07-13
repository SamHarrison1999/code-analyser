import ast
from abc import ABC, abstractmethod

class ASTMetricPlugin(ABC):
    """
    Abstract base class for AST metric plugins.

    Each plugin must:
    - Provide a unique metric name via the `name()` method.
    - Implement `visit()` to compute the metric using an AST and source code.
    """

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
