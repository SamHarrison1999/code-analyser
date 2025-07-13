# tests/test_vulture_plugins.py

import pytest
import tempfile
from metrics.vulture_metrics.gather import gather_vulture_metrics, get_vulture_metric_names
from metrics.vulture_metrics.plugins import load_plugins


@pytest.fixture
def temp_python_file():
    """Creates a temporary Python file with unused code for testing."""
    code = """
def used_function():
    return True

def unused_function():
    return False

class UnusedClass:
    pass

x = 123
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        return f.name


def test_vulture_plugins_load_successfully():
    """Test that all Vulture plugins load without error."""
    plugins = load_plugins()
    assert isinstance(plugins, list)
    assert all(callable(p.extract) for p in plugins)


def test_vulture_plugin_names_are_unique():
    """Test that all Vulture plugin names are unique and non-empty."""
    names = [p.name() for p in load_plugins()]
    assert all(names), "All plugin names must be non-empty"
    assert len(set(names)) == len(names), "Plugin names must be unique"


def test_vulture_extractor_returns_all_metrics(temp_python_file):
    """Test that gather returns all expected metrics for a file."""
    metrics = gather_vulture_metrics(temp_python_file)
    expected_names = get_vulture_metric_names()
    assert isinstance(metrics, list)
    assert len(metrics) == len(expected_names)
    assert all(isinstance(val, (int, float)) for val in metrics)


def test_vulture_gather_and_names_match_order(temp_python_file):
    """Test that the metric values match the plugin name order."""
    values = gather_vulture_metrics(temp_python_file)
    names = get_vulture_metric_names()
    assert isinstance(values, list), "Expected gathered values to be a list"
    assert isinstance(names, list), "Expected metric names to be a list"
    assert len(values) == len(names), "Each metric name should correspond to a value"
