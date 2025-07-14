# ✅ SonarQube plugin system interface
# ⚙️ Provides gatherer function and introspection for all registered metric plugins

from typing import Callable, Type
from metrics.sonar_metrics.gather import gather_sonar_metrics
from metrics.sonar_metrics.plugins.base import SonarMetricPlugin
from metrics.sonar_metrics.plugins import load_plugins

# ✅ Dynamically load all sonar plugins once to ensure consistency
plugins: list[SonarMetricPlugin] = load_plugins()

# ✅ Returns the callable that aggregates and returns all sonar metrics as a flat dict
def get_metric_gatherer() -> Callable[[str], dict[str, float]]:
    return gather_sonar_metrics

# ✅ Returns fully qualified metric names for all loaded SonarQube plugins
def get_metric_names() -> list[str]:
    return [plugin.name() for plugin in plugins]  # ❗ Drop "sonar." prefix for cleaner flat key usage

# ✅ Useful for testing, inspection, or dynamic plugin filtering
def get_metric_classes() -> list[Type[SonarMetricPlugin]]:
    return [plugin.__class__ for plugin in plugins]
