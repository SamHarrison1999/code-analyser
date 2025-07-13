import pytest
from metrics.cloc_metrics.plugins import load_plugins
from metrics.cloc_metrics.extractor import ClocExtractor
from metrics.cloc_metrics.gather import gather_cloc_metrics, get_cloc_metric_names

@pytest.fixture
def cloc_plugins():
    return load_plugins()

def test_cloc_plugins_load_successfully(cloc_plugins):
    assert isinstance(cloc_plugins, list)
    assert all(callable(getattr(p, "extract", None)) for p in cloc_plugins)
    assert all(callable(getattr(p, "name", None)) for p in cloc_plugins)

def test_cloc_plugin_names_are_unique(cloc_plugins):
    names = [p.name() for p in cloc_plugins]
    assert all(name.strip() for name in names), "Plugin names must not be empty"
    assert len(set(names)) == len(names), "Plugin names must be unique"

def test_cloc_extractor_returns_all_metrics(tmp_path):
    test_file = tmp_path / "sample.py"
    test_file.write_text('''# comment\nprint("Hello")\n\n# another comment''')

    extractor = ClocExtractor(str(test_file))
    results = extractor.extract()

    assert isinstance(results, dict)
    assert all(isinstance(v, (int, float)) for v in results.values())

def test_cloc_gather_and_names_match_order(tmp_path):
    test_file = tmp_path / "example.py"
    test_file.write_text("print('hello')\n# comment")

    metrics = gather_cloc_metrics(str(test_file))
    names = get_cloc_metric_names()

    assert isinstance(metrics, list)
    assert isinstance(names, list)
    assert len(metrics) == len(names)
    assert all(isinstance(m, (int, float)) for m in metrics)
