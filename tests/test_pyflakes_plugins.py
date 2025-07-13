import pytest
from metrics.pyflakes_metrics import load_plugins
from metrics.pyflakes_metrics.plugins.base import PyflakesMetricPlugin
from metrics.pyflakes_metrics.extractor import PyflakesExtractor
from metrics.pyflakes_metrics.gather import gather_pyflakes_metrics, get_pyflakes_metric_names


@pytest.fixture
def pyflakes_plugins():
    return load_plugins()


def test_pyflakes_plugins_load_successfully(pyflakes_plugins):
    assert all(isinstance(p, PyflakesMetricPlugin) for p in pyflakes_plugins)
    assert len(pyflakes_plugins) > 0


def test_pyflakes_plugin_names_are_unique(pyflakes_plugins):
    names = [p.name() for p in pyflakes_plugins]
    assert len(set(names)) == len(names), "Plugin names must be unique"
    assert all(name.strip() for name in names)


def test_pyflakes_extractor_returns_all_metrics(pyflakes_plugins, tmp_path):
    file_path = tmp_path / "sample.py"
    file_path.write_text("x = y\n")

    extractor = PyflakesExtractor(str(file_path))
    output = extractor.extract()

    for plugin in pyflakes_plugins:
        metric = plugin.extract(output, str(file_path))
        assert isinstance(metric, (int, float))


def test_pyflakes_gather_and_names_match_order(tmp_path):
    file_path = tmp_path / "sample.py"
    file_path.write_text("x = y\n")

    values = gather_pyflakes_metrics(str(file_path))
    names = get_pyflakes_metric_names()

    assert isinstance(values, list)
    assert all(isinstance(v, (int, float)) for v in values)
    assert len(values) == len(names)
