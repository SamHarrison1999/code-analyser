# File: tests/test_plugins.py

import ast
from metrics.ast_metrics.plugins.default_plugins import (
    ModuleDocstringPlugin,
    TodoCommentPlugin,
    NestedFunctionPlugin,
)

# Sample code snippets for testing
CODE_WITH_MODULE_DOCSTRING = '"""This is a module docstring."""\n\ndef foo():\n    pass\n'
CODE_WITH_TODO = '# TODO: Refactor this function\n\ndef foo():\n    pass\n'
CODE_WITH_NESTED_FUNCTION = (
    "def outer():\n"
    "    def inner():\n"
    "        return 42\n"
)

def test_module_docstring_plugin():
    """Test that the ModuleDocstringPlugin detects top-level docstrings."""
    tree = ast.parse(CODE_WITH_MODULE_DOCSTRING)
    plugin = ModuleDocstringPlugin()
    assert plugin.visit(tree, CODE_WITH_MODULE_DOCSTRING) == 1

def test_todo_comment_plugin():
    """Test that the TodoCommentPlugin counts TODO comments correctly."""
    tree = ast.parse(CODE_WITH_TODO)
    plugin = TodoCommentPlugin()
    assert plugin.visit(tree, CODE_WITH_TODO) == 1

def test_nested_function_plugin():
    """Test that the NestedFunctionPlugin detects nested function definitions."""
    tree = ast.parse(CODE_WITH_NESTED_FUNCTION)
    plugin = NestedFunctionPlugin()
    assert plugin.visit(tree, CODE_WITH_NESTED_FUNCTION) == 1
