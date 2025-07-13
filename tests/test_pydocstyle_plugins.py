import pytest
from metrics.pydocstyle_metrics.plugins import load_plugins
from metrics.pydocstyle_metrics.gather import get_pydocstyle_metric_names
from metrics.pydocstyle_metrics.extractor import PydocstyleExtractor

@pytest.fixture
def example_output():
    return [
        "example.py:1 in public module\n        D100: Missing docstring in public module",
        "example.py:5 in public function\n        D103: Missing docstring in public function",
        "example.py:10 in public class\n        D101: Missing docstring in public class",
    ]

def test_plugins_load_successfully():
    plugins = load_plugins()
    assert isinstance(plugins, list)
    assert all(hasattr(p, "extract") for p in plugins)
    assert all(callable(p.extract) for p in plugins)

def test_plugin_names_are_unique():
    plugins = load_plugins()
    names = [p.name() for p in plugins]
    assert len(names) == len(set(names)), "Plugin names must be unique"

def test_get_metric_names_matches_plugins():
    plugins = load_plugins()
    names_from_plugins = [p.name() for p in plugins]
    names_from_loader = get_pydocstyle_metric_names()
    assert names_from_plugins == names_from_loader

def test_plugin_outputs_match_expected(example_output):
    plugins = load_plugins()
    file_path = "example.py"
    results = {plugin.name(): plugin.extract(example_output, file_path) for plugin in plugins}

    assert isinstance(results["number_of_pydocstyle_violations"], int)
    assert isinstance(results["number_of_missing_doc_strings"], int)
    assert isinstance(results["percentage_of_compliance_with_docstring_style"], float)
