import pytest
from metrics.bandit_metrics.plugins import load_plugins
from metrics.bandit_metrics.extractor import BanditExtractor
from metrics.bandit_metrics.gather import gather_bandit_metrics, get_bandit_metric_names

def test_bandit_plugins_load_successfully():
    """
    Ensure all Bandit plugins are discovered and instantiated correctly.
    """
    plugins = load_plugins()
    assert plugins, "No Bandit plugins loaded"
    assert all(callable(getattr(p, "extract", None)) for p in plugins), "Some plugins missing 'extract' method"
    assert all(isinstance(p.name(), str) and p.name() for p in plugins), "Each plugin must have a non-empty string name"

def test_bandit_extractor_metrics_are_numeric(tmp_path):
    """
    Run BanditExtractor and verify all metrics are numeric values.
    """
    test_file = tmp_path / "sample.py"
    test_file.write_text("""
    import os

    def insecure():
        os.system("ls")  # Bandit B605
    """)

    extractor = BanditExtractor(str(test_file))
    result = extractor.extract()
    assert isinstance(result, dict), "Result should be a dictionary"
    assert all(isinstance(v, (int, float)) for v in result.values()), "All plugin values must be numeric"

def test_bandit_gather_and_names_match_order(tmp_path):
    """
    Check that gather_bandit_metrics and get_bandit_metric_names align in length and order.
    """
    test_file = tmp_path / "sample.py"
    test_file.write_text("print('Hello world')")

    values = gather_bandit_metrics(str(test_file))
    names = get_bandit_metric_names()

    assert isinstance(values, list)
    assert isinstance(names, list)
    assert len(values) == len(names), f"Mismatch: {len(values)} values vs {len(names)} names"
    assert all(isinstance(v, (int, float)) for v in values), "All gathered values must be numeric"
