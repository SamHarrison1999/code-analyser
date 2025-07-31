# File: code_analyser/src/metrics/lizard_metrics/plugins/__init__.py

import pkgutil
import importlib
import inspect
import os
import logging

from .base import LizardMetricPlugin as BasePlugin

logger = logging.getLogger(__name__)
__all__ = []
_discovered_plugins = {}

# ✅ Best Practice: Dynamic plugin registration with introspection support
# ⚠️ SAST Risk: Plugin loading must be guarded with try/except for safe runtime fallback

# 🔍 Discover all plugin modules except base.py
for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name == "base":
        continue
    try:
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
                logger.debug(
                    f"✅ Loaded Lizard plugin: {plugin_id} (tags={plugin_tags})"
                )
    except Exception as e:
        logger.warning(
            f"⚠️ Failed to load Lizard plugin module '{module_name}': {type(e).__name__}: {e}"
        )

# ✅ Alias for plugin base class
LizardMetricPlugin = BasePlugin


def load_plugins() -> list[BasePlugin]:
    """Instantiate and return all discovered Lizard metric plugin classes."""
    return [entry["class"]() for entry in _discovered_plugins.values()]


def load_plugins_by_tag(tag: str) -> list[BasePlugin]:
    """Return Lizard plugins matching a specific tag."""
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
    """Return plugin metadata (name, tags) for all registered plugins."""
    return [
        {"name": entry["name"], "tags": entry["tags"]}
        for entry in _discovered_plugins.values()
    ]
