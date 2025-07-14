import pytest
from metrics.sonar_metrics.plugins import load_plugins


@pytest.mark.parametrize("plugin", load_plugins())
def test_sonar_plugin_extract(plugin):
    dummy_context = {plugin.name(): 42.0}
    result = plugin.extract(dummy_context, "fake/path.py")
    assert isinstance(result, float)
    assert result >= 0

