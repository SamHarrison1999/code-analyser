# File: code_analyser/src/metrics/bandit_metrics/plugins/__init__.py

import pkgutil
import importlib
import inspect
import os

from .base import BanditMetricPlugin as BasePlugin

# ✅ Best Practice: Registry and metadata tracking for plugin discovery and filtering
__all__ = []
_discovered_plugins = {}

# ✅ Plugin auto-discovery
for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name == "base":
        continue
    module = importlib.import_module(f"{__name__}.{module_name}")
    for name, obj in inspect.getmembers(module):
        # 🧠 ML Signal: Dynamically loading plugin class hierarchy allows feature injection
        # ⚠️ SAST Risk: Restrict to only expected base class subclasses
        if inspect.isclass(obj) and issubclass(obj, BasePlugin) and obj is not BasePlugin:
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
BanditMetricPlugin = BasePlugin


def load_plugins() -> list[BasePlugin]:
    """Return all discovered BanditMetricPlugin instances."""
    return [entry["class"]() for entry in _discovered_plugins.values()]


def load_plugins_by_tag(tag: str) -> list[BasePlugin]:
    """Return all plugins tagged with the given label (e.g., 'security')."""
    return [entry["class"]() for entry in _discovered_plugins.values() if tag in entry["tags"]]


def get_plugin_by_name(name: str) -> BasePlugin | None:
    """Retrieve a plugin instance by its name."""
    entry = _discovered_plugins.get(name)
    return entry["class"]() if entry else None


def list_plugins_metadata() -> list[dict]:
    """Return metadata for all registered plugins."""
    return [
        {"name": entry["name"], "tags": entry["tags"]} for entry in _discovered_plugins.values()
    ]
