import pytest
import tempfile
from pathlib import Path
from metrics.pylint_metrics.plugins import load_plugins
from metrics.pylint_metrics.extractor import PylintMetricExtractor
from metrics.pylint_metrics.gather import gather_pylint_metrics, get_pylint_metric_names

@pytest.fixture
def temp_python_file():
    code = """
def example_function():
    print('Hello')
    return 1
"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as f:
        f.write(code)
        f.flush()
        yield Path(f.name)

def test_pylint_plugins_load_successfully():
    plugins = load_plugins()
    assert plugins, "No Pylint plugins loaded"
    assert all(callable(p.extract) for p in plugins)

def test_pylint_plugin_names_are_unique():
    plugins = load_plugins()
    names = [p.name() for p in plugins]
    assert len(set(names)) == len(names), "Plugin names must be unique"

def test_pylint_extractor_returns_all_metrics(temp_python_file):
    extractor = PylintMetricExtractor(str(temp_python_file))
    metrics = extractor.extract()
    expected_keys = [p.name() for p in load_plugins()]
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == set(expected_keys)
    assert all(isinstance(v, (int, float)) for v in metrics.values())

def test_pylint_gather_and_names_match_order(temp_python_file):
    values = gather_pylint_metrics(str(temp_python_file))
    names = get_pylint_metric_names()
    assert isinstance(values, list), "Expected gathered values to be a list"
    assert len(values) == len(names)
    assert all(isinstance(v, (int, float)) for v in values)
