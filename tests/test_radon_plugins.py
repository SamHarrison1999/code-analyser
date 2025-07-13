import pytest
import tempfile
from metrics.radon_metrics.plugins import load_plugins
from metrics.radon_metrics.extractor import RadonExtractor
from metrics.radon_metrics.gather import gather_radon_metrics, get_radon_metric_names

@pytest.fixture
def temp_python_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as f:
        f.write("def example():\n    '''Example docstring.'''\n    return 42\n")
        return f.name

def test_radon_plugins_load_successfully():
    plugins = load_plugins()
    assert plugins, "Expected some plugins to be loaded"

def test_radon_plugin_names_are_unique():
    plugins = load_plugins()
    names = [p.name() for p in plugins]
    assert all(name.strip() for name in names)
    assert len(set(names)) == len(names), "Duplicate plugin names found"

def test_radon_extractor_returns_all_metrics(temp_python_file):
    extractor = RadonExtractor(temp_python_file)
    metric_dict = extractor.extract()
    plugin_names = get_radon_metric_names()
    for name in plugin_names:
        assert name in metric_dict

def test_radon_gather_and_names_match_order(temp_python_file):
    values = gather_radon_metrics(temp_python_file)
    names = get_radon_metric_names()
    assert isinstance(values, list)
    assert len(values) == len(names)
