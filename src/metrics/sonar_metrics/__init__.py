# ✅ Structured SonarQube metrics package interface

# ✅ Best Practice: Explicit relative imports make module relationships clear within a package
from .extractor import (
    get_metric_gatherer,
    get_metric_names,
    get_metric_classes,
    load_plugins
)
from .gather import gather_sonar_metrics

# ✅ Best Practice: Define public API clearly for cleaner external access
__all__ = [
    "get_metric_gatherer",
    "get_metric_names",
    "get_metric_classes",
    "gather_sonar_metrics",
    "load_plugins"
]
