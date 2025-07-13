import pytest
from metrics.flake8_metrics.plugins import load_plugins
from metrics.flake8_metrics.extractor import Flake8Extractor
from metrics.flake8_metrics.gather import gather_flake8_metrics, get_flake8_metric_names

@pytest.fixture
def flake8_plugins():
    return load_plugins()

def test_flake8_plugins_load_successfully(flake8_plugins):
    assert len(flake8_plugins) > 0
    for plugin in flake8_plugins:
        assert hasattr(plugin, 'name')
        assert callable(plugin.name)
        assert hasattr(plugin, 'extract')
        assert callable(plugin.extract)

def test_flake8_plugin_names_are_unique(flake8_plugins):
    names = [p.name() for p in flake8_plugins]
    assert len(set(names)) == len(names), "Plugin names must be unique"

def test_flake8_extractor_returns_all_metrics(tmp_path):
    code = "x = 1\nprint(x)\n"
    test_file = tmp_path / "example.py"
    test_file.write_text(code)

    extractor = Flake8Extractor(str(test_file))
    metrics = extractor.extract()
    assert isinstance(metrics, dict)
    for name in get_flake8_metric_names():
        assert name in metrics

def test_flake8_gather_and_names_match_order(tmp_path):
    test_file = tmp_path / "sample.py"
    test_file.write_text("import os\nprint(os)\n")

    values = gather_flake8_metrics(str(test_file))
    names = get_flake8_metric_names()

    assert isinstance(values, list)
    assert isinstance(names, list)
    assert len(values) == len(names)
    assert all(isinstance(v, (int, float)) for v in values)
