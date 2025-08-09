# File: code_analyser/src/metrics/ast_metrics/extractor.py

import ast
import logging
from typing import Dict, List, Optional

from metrics.ast_metrics.plugins import ASTMetricPlugin, load_plugins

logger = logging.getLogger(__name__)


class ASTMetricExtractor:
    """
    Extracts AST-based static metrics from a Python source file.

    Combines pluggable ASTMetricPlugin instances with structural AST traversal
    to produce a comprehensive dictionary of code metrics.
    """

    def __init__(self, file_path: str, plugins: Optional[List[ASTMetricPlugin]] = None) -> None:
        self.file_path: str = file_path
        self.plugins: List[ASTMetricPlugin] = plugins if plugins is not None else load_plugins()
        self.metrics: Dict[str, int] = self._init_metrics()
        self.code: str = ""
        self.tree: Optional[ast.AST] = None

    def extract(self) -> Dict[str, int]:
        """
        Perform the complete metric extraction pipeline: parse source,
        apply plugins, and collect structural metrics.

        Returns:
            Dict[str, int]: Mapping of metric names to computed values.
        """
        if not self._read_file() or not self._parse_ast():
            return self.metrics

        self._apply_plugins()
        self._walk_ast()
        return self.metrics

    def get_confidences(self) -> Dict[str, float]:
        """
        Return a mapping of metric names to confidence scores (0.0–1.0).

        Returns:
            Dict[str, float]
        """
        if not self.tree:
            return {}
        return {
            plugin.name(): plugin.confidence_score(self.tree, self.code) for plugin in self.plugins
        }

    def get_severities(self) -> Dict[str, str]:
        """
        Return a mapping of metric names to severity levels (low, medium, high).

        Returns:
            Dict[str, str]
        """
        if not self.tree:
            return {}
        return {
            plugin.name(): plugin.severity_level(self.tree, self.code) for plugin in self.plugins
        }

    def _read_file(self) -> bool:
        """
        Read the source file into memory.

        Returns:
            bool: True if read successfully, False otherwise.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.code = f.read()
            return True
        except Exception as e:
            logging.error(f"[ASTMetricExtractor] Failed to read file '{self.file_path}': {e}")
            return False

    def _parse_ast(self) -> bool:
        """
        Parse the source code into an AST.

        Returns:
            bool: True if parsing succeeds, False otherwise.
        """
        try:
            self.tree = ast.parse(self.code)
            return True
        except Exception as e:
            logging.error(f"[ASTMetricExtractor] AST parsing failed: {e}")
            return False

    def _apply_plugins(self) -> None:
        """
        Apply all loaded ASTMetricPlugin instances to extract metric values.
        """
        for plugin in self.plugins:
            try:
                result = plugin.visit(self.tree, self.code)
                self.metrics[plugin.name()] = result
                logger.debug(f"[ASTMetricExtractor] {plugin.name()} = {result}")
            except Exception as e:
                logging.warning(
                    f"[ASTMetricExtractor] Plugin '{plugin.name()}' failed: {e}",
                    exc_info=True,
                )

    def _walk_ast(self) -> None:
        """
        Traverse the AST to collect structural (non-plugin) metrics.
        """
        for node in ast.walk(self.tree):
            self._handle_node(node)

    def _handle_node(self, node: ast.AST) -> None:
        """
        Handle an individual AST node and update structural metrics.

        Args:
            node (ast.AST): A single node from the AST.
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.metrics["functions"] += 1
            if ast.get_docstring(node):
                self.metrics["function_docstrings"] += 1
            if node.name.startswith("__") and node.name.endswith("__"):
                self.metrics["magic_methods"] += 1
            self.metrics["lambda_functions"] += sum(
                isinstance(n, ast.Lambda) for n in ast.walk(node)
            )
            self.metrics["chained_methods"] += sum(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Attribute)
                for n in ast.walk(node)
            )

        elif isinstance(node, ast.ClassDef):
            self.metrics["classes"] += 1
            if ast.get_docstring(node):
                self.metrics["class_docstrings"] += 1

        elif isinstance(node, ast.Assert):
            self.metrics["assert_statements"] += 1

        elif isinstance(node, ast.ExceptHandler):
            self.metrics["exceptions"] += 1

        elif isinstance(node, (ast.For, ast.While, ast.If)):
            self.metrics["loops_conditionals"] += 1

        elif isinstance(node, ast.Global):
            self.metrics["global_variables"] += len(node.names)

        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            self.metrics["chained_methods"] += 1

    def _init_metrics(self) -> Dict[str, int]:
        """
        Initialise the metric dictionary with zero values for all expected metrics.

        Returns:
            Dict[str, int]: Dictionary of metric names initialised to zero.
        """
        return {
            # Plugin-based metrics
            "module_docstring": 0,
            "todo_comments": 0,
            "nested_functions": 0,
            "lambda_functions": 0,
            "magic_methods": 0,
            "assert_statements": 0,
            "class_docstrings": 0,
            # Core structural metrics
            "functions": 0,
            "classes": 0,
            "function_docstrings": 0,
            "exceptions": 0,
            "loops_conditionals": 0,
            "global_variables": 0,
            "chained_methods": 0,
        }
