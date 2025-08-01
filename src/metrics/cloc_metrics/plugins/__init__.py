# File: code_analyser/src/metrics/cloc_metrics/plugins/__init__.py

import pkgutil
import importlib
import inspect
import os

from .base import ClocMetricPlugin as BasePlugin

# ✅ Best Practice: Dynamic plugin registry
__all__ = []
_discovered_plugins = {}

# 🧠 ML Signal: Loading plugins dynamically allows LoC metrics to evolve with the tool
for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name == "base":
        continue
    module = importlib.import_module(f"{__name__}.{module_name}")
    for name, obj in inspect.getmembers(module):
        if (
            inspect.isclass(obj)
            and issubclass(obj, BasePlugin)
            and obj is not BasePlugin
        ):
            globals()[name] = obj
            __all__.append(name)
            plugin_id = getattr(obj, "plugin_name", name)
            plugin_tags = getattr(obj, "plugin_tags", [])
            _discovered_plugins[plugin_id] = {
                "class": obj,
                "name": plugin_id,
                "tags": plugin_tags,
            }

# ✅ Aliased export for base class
ClocMetricPlugin = BasePlugin


def load_plugins() -> list[BasePlugin]:
    """Return all discovered ClocMetricPlugin instances."""
    return [entry["class"]() for entry in _discovered_plugins.values()]


def load_plugins_by_tag(tag: str) -> list[BasePlugin]:
    """Return all plugins with the given tag (e.g., 'blank', 'comment')."""
    return [
        entry["class"]()
        for entry in _discovered_plugins.values()
        if tag in entry["tags"]
    ]


def get_plugin_by_name(name: str) -> BasePlugin | None:
    """Retrieve a plugin instance by its unique name."""
    entry = _discovered_plugins.get(name)
    return entry["class"]() if entry else None


def list_plugins_metadata() -> list[dict]:
    """Return metadata about all registered plugins."""
    return [
        {"name": entry["name"], "tags": entry["tags"]}
        for entry in _discovered_plugins.values()
    ]
