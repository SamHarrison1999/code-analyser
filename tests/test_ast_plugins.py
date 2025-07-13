import pytest
from metrics.ast_metrics.plugins import load_plugins

@pytest.fixture(scope="module")
def loaded_plugins():
    return load_plugins()

def test_plugins_are_instantiated(loaded_plugins):
    """
    All loaded plugins should be instances of ASTMetricPlugin subclasses.
    """
    from metrics.ast_metrics.plugins.base import ASTMetricPlugin
    assert all(isinstance(p, ASTMetricPlugin) for p in loaded_plugins)

def test_plugin_names_are_unique_and_nonempty(loaded_plugins):
    """
    Plugin names must be unique and not empty.
    """
    names = [p.name() for p in loaded_plugins]
    assert all(isinstance(n, str) and n.strip() for n in names), "Each plugin must have a non-empty name"
    assert len(names) == len(set(names)), "Plugin names must be unique"

def test_expected_plugins_are_loaded(loaded_plugins):
    """
    Check that key expected plugins are present by name.
    """
    expected = {
        "module_docstring",
        "todo_comments",
        "nested_functions",
        "lambda_functions",
        "magic_methods",
        "assert_statements",
        "class_docstrings",
        "functions",
        "classes",
        "function_docstrings",
        "exceptions",
        "loops_conditionals",
        "global_variables",
        "chained_methods",
    }
    actual = {p.name() for p in loaded_plugins}
    missing = expected - actual
    assert not missing, f"Missing expected plugins: {missing}"
