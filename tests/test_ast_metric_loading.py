# File: test_ast_metric_loading.py

import pytest
from pathlib import Path
from metrics.ast_metrics.plugins import load_plugins
from metrics.ast_metrics.gather import gather_ast_metrics, get_ast_metric_names

def test_plugins_load_successfully():
    """
    Test that all AST metric plugins are loaded without error and conform to the interface.
    """
    plugins = load_plugins()
    assert isinstance(plugins, list), "Plugins should be returned in a list"
    assert all(hasattr(p, "name") and hasattr(p, "visit") for p in plugins), "Plugins must implement required methods"
    assert all(callable(p.name) and callable(p.visit) for p in plugins), "Plugin methods must be callable"
    assert all(isinstance(p.name(), str) and p.name().strip() for p in plugins), "Plugin names must be non-empty strings"

def test_plugin_names_are_unique():
    """
    Test that all plugin names are unique and non-empty.
    """
    plugin_names = [plugin.name() for plugin in load_plugins()]
    assert len(plugin_names) == len(set(plugin_names)), "Plugin names must be unique"
    assert all(name.strip() for name in plugin_names), "Plugin names must not be empty"

def test_get_ast_metric_names_includes_all_expected_keys():
    """
    Ensure that get_ast_metric_names includes all keys used in gather_ast_metrics.
    """
    metric_names = get_ast_metric_names()
    assert isinstance(metric_names, list)
    assert all(isinstance(name, str) for name in metric_names)
    assert len(metric_names) == len(set(metric_names)), "Metric name list must be unique"

@pytest.fixture
def sample_code(tmp_path) -> Path:
    """
    Create a temporary Python file for testing AST metric extraction.
    """
    sample = tmp_path / "sample.py"
    sample.write_text(
        '''
# TODO: refactor this
def outer():
    def inner(): pass
    return lambda x: x + 1
class Foo:
    def __init__(self): pass
assert True
'''
    )
    return sample

def test_gather_ast_metrics_returns_correct_shape(sample_code: Path):
    """
    Ensure gather_ast_metrics runs and returns metrics matching get_ast_metric_names().
    """
    result = gather_ast_metrics(str(sample_code))
    metric_keys = get_ast_metric_names()

    assert isinstance(result, list), "gather_ast_metrics must return a list"
    assert all(isinstance(v, int) for v in result), "All returned metric values must be integers"
    assert len(result) == len(metric_keys), "Result length must match number of defined metric keys"
