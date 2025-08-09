# File: code_analyser/src/metrics/sonar_metrics/plugins/__init__.py

"""
SonarQube Metric Plugin Loader

Dynamically discovers and loads SonarMetricPlugin subclasses from this directory.

Features:
- Dynamic plugin auto-discovery
- Tag and name-based plugin filtering
- Unified metadata access for ML/GUI pipelines
"""

import pkgutil
import importlib
import inspect
import os
import logging

from .base import SonarMetricPlugin as BasePlugin

logger = logging.getLogger(__name__)
__all__ = []
_discovered_plugins = {}

# ✅ Best Practice: Centralise plugin metadata and enable tag-based filtering
# ⚠️ SAST Risk: Skip invalid or base-only modules safely
# 🧠 ML Signal: Registry defines SonarQube-derived metrics for supervised learning

plugin_dir = os.path.dirname(__file__)
for _, module_name, _ in pkgutil.iter_modules([plugin_dir]):
    if module_name.startswith("_") or module_name == "base":
        continue
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                globals()[name] = obj
                __all__.append(name)
                plugin_id = getattr(obj, "plugin_name", name)
                plugin_tags = getattr(obj, "plugin_tags", [])
                _discovered_plugins[plugin_id] = {
                    "class": obj,
                    "name": plugin_id,
                    "tags": plugin_tags,
                }
                logger.debug(f"✅ Loaded Sonar plugin: {plugin_id} (tags={plugin_tags})")
    except Exception as e:
        logger.warning(
            f"⚠️ Failed to load Sonar plugin module '{module_name}': {type(e).__name__}: {e}"
        )

# Alias for external use
SonarMetricPlugin = BasePlugin


def load_plugins() -> list[BasePlugin]:
    """Instantiate and return all discovered Sonar metric plugins."""
    return [entry["class"]() for entry in _discovered_plugins.values()]


def load_plugins_by_tag(tag: str) -> list[BasePlugin]:
    """Return only Sonar plugins tagged with a specific label."""
    return [entry["class"]() for entry in _discovered_plugins.values() if tag in entry["tags"]]


def get_plugin_by_name(name: str) -> BasePlugin | None:
    """Return a plugin instance by its unique name."""
    entry = _discovered_plugins.get(name)
    return entry["class"]() if entry else None


def list_plugins_metadata() -> list[dict]:
    """Return metadata about all registered Sonar plugins."""
    return [
        {"name": entry["name"], "tags": entry["tags"]} for entry in _discovered_plugins.values()
    ]
