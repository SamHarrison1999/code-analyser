"""
Unit tests for AST metric plugins.

Covers:
- Module docstring detection
- TODO/FIXME comments
- Nested function detection
"""

import ast
from metrics.ast_metrics.plugins.default_plugins import (
    ModuleDocstringPlugin,
    TodoCommentPlugin,
    NestedFunctionPlugin,
)


def test_module_docstring_plugin():
    code = '"""A docstring."""\ndef func(): pass'
    tree = ast.parse(code)
    assert ModuleDocstringPlugin().visit(tree, code) == 1

    no_doc_code = "def func(): pass"
    tree2 = ast.parse(no_doc_code)
    assert ModuleDocstringPlugin().visit(tree2, no_doc_code) == 0


def test_todo_comment_plugin():
    code = """
# TODO: fix this
# nothing here
# FIXME: needs work
# random
    """
    tree = ast.parse(code)
    assert TodoCommentPlugin().visit(tree, code) == 2


def test_nested_function_plugin():
    code = """
def outer():
    def inner():
        def deeper():
            return True
        return deeper()
    return inner()
    """
    tree = ast.parse(code)
    assert NestedFunctionPlugin().visit(tree, code) == 2  # inner, deeper
