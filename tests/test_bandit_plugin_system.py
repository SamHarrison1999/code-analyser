import pytest
from metrics.bandit_metrics.plugins import load_plugins, BanditMetricPlugin
from metrics.bandit_metrics.extractor import BanditExtractor

@pytest.fixture
def bandit_plugins():
    return load_plugins()

def test_bandit_plugins_load_correctly(bandit_plugins):
    """
    Ensure BanditMetricPlugin subclasses are dynamically loaded.
    """
    assert isinstance(bandit_plugins, list)
    assert all(isinstance(p, BanditMetricPlugin) for p in bandit_plugins)
    assert len(bandit_plugins) > 0, "No plugins were loaded"

def test_bandit_plugin_names_unique(bandit_plugins):
    names = [p.name() for p in bandit_plugins]
    assert all(name.strip() for name in names)
    assert len(set(names)) == len(names), "Plugin names must be unique"

def test_bandit_extractor_returns_all_metrics(tmp_path):
    """
    Check that BanditExtractor returns all plugin metric keys, even if Bandit finds no issues.
    """
    file_path = tmp_path / "example.py"
    file_path.write_text("def harmless():\n    return 1\n")

    extractor = BanditExtractor(str(file_path))
    metrics = extractor.extract()

    plugin_names = [p.name() for p in extractor.plugins]

    assert isinstance(metrics, dict)
    assert all(name in metrics for name in plugin_names)
    assert all(isinstance(metrics[name], (int, float)) for name in plugin_names)
