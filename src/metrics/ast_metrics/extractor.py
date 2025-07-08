# File: src/metrics/ast_metrics/extractor.py

import ast
import logging
from typing import Dict, List

from metrics.ast_metrics.plugins import ASTMetricPlugin, load_plugins


class ASTMetricExtractor:
    """
    Extracts AST-based static metrics from a Python source file.

    This extractor loads pluggable AST metric plugins and combines them with
    additional AST node pattern analysis (e.g. function and class counting).
    """

    def __init__(self, file_path: str, plugins: List[ASTMetricPlugin] = None) -> None:
        self.file_path = file_path
        self.plugins = plugins if plugins else load_plugins()
        self.metrics = self._init_metrics()
        self.code = ""
        self.tree = None

    def extract(self) -> Dict[str, int]:
        """
        Executes the full AST parsing and metric extraction process.

        Returns:
            dict[str, int]: A dictionary of metric names and their values.
        """
        if not self._read_file() or not self._parse_ast():
            return self.metrics
        self._apply_plugins()
        self._walk_ast()
        return self.metrics

    def _read_file(self) -> bool:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.code = f.read()
            return True
        except Exception as e:
            logging.error(f"[ASTExtractor] Failed to read file '{self.file_path}': {e}")
            return False

    def _parse_ast(self) -> bool:
        try:
            self.tree = ast.parse(self.code)
            return True
        except Exception as e:
            logging.error(f"[ASTExtractor] AST parse failed: {e}")
            return False

    def _apply_plugins(self) -> None:
        for plugin in self.plugins:
            try:
                self.metrics[plugin.name()] = plugin.visit(self.tree, self.code)
            except Exception as e:
                logging.warning(f"[ASTExtractor] Plugin '{plugin.name()}' failed: {e}")

    def _walk_ast(self) -> None:
        for node in ast.walk(self.tree):
            self._handle_node(node)

    def _handle_node(self, node: ast.AST) -> None:
        """
        Collects non-plugin metrics from raw AST traversal.
        These supplement the plugin results.
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.metrics["functions"] += 1
            if ast.get_docstring(node):
                self.metrics["function_docstrings"] += 1
            if node.name.startswith("__") and node.name.endswith("__"):
                self.metrics["magic_methods"] += 1
            self.metrics["lambda_functions"] += sum(isinstance(n, ast.Lambda) for n in ast.walk(node))
            self.metrics["chained_methods"] += sum(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Attribute)
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
        Initializes the base metric dictionary with zeroed counters.
        This includes both plugin-populated and core metrics.
        """
        return {
            "functions": 0,
            "classes": 0,
            "function_docstrings": 0,
            "class_docstrings": 0,
            "module_docstring": 0,
            "todo_comments": 0,
            "assert_statements": 0,
            "exceptions": 0,
            "loops_conditionals": 0,
            "nested_functions": 0,
            "global_variables": 0,
            "chained_methods": 0,
            "magic_methods": 0,
            "lambda_functions": 0,
        }
