import pytest
from metrics.lizard_metrics.plugins import load_plugins
from metrics.lizard_metrics.extractor import extract_lizard_metrics
from metrics.lizard_metrics.gather import gather_lizard_metrics, get_lizard_metric_names

TEST_FILE = "test_lizard_sample.py"

def test_lizard_plugins_load_successfully():
    plugins = load_plugins()
    assert plugins, "No Lizard plugins loaded"
    assert all(callable(p.extract) for p in plugins), "Each plugin must have extract()"
    assert all(callable(p.name) for p in plugins), "Each plugin must have name()"

def test_lizard_plugin_names_are_unique():
    names = [p.name() for p in load_plugins()]
    assert all(name.strip() for name in names), "Plugin names must not be empty"
    assert len(set(names)) == len(names), "Plugin names must be unique"

def test_lizard_extractor_returns_all_metrics():
    metrics = extract_lizard_metrics(TEST_FILE)
    assert isinstance(metrics, dict)
    assert all(isinstance(v, (int, float)) for v in metrics.values()), "All Lizard metrics should be numeric"

def test_lizard_gather_and_names_match_order():
    values = gather_lizard_metrics(TEST_FILE)
    names = get_lizard_metric_names()
    assert isinstance(values, list)
    assert isinstance(names, list)
    assert len(values) == len(names), "Mismatch in number of metric values and names"
    assert all(isinstance(v, (int, float)) for v in values), "All gathered metrics should be numeric"
