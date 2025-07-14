import logging
import os
import pkgutil
import importlib
import inspect
from .base import SonarMetricPlugin as BasePlugin

# ✅ Alias for external reference consistency
SonarMetricPlugin = BasePlugin

__all__ = []

# 📁 Path to the current plugin directory
plugin_dir = os.path.dirname(__file__)

# 🔍 Discover and import all plugin classes in this directory
for _, module_name, _ in pkgutil.iter_modules([plugin_dir]):
    if module_name.startswith("_") or module_name == "base":
        continue  # 🚫 Skip private or base definition modules

    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                globals()[name] = obj
                __all__.append(name)
    except Exception as e:
        logging.warning(f"[SonarPluginLoader] Skipped {module_name}: {type(e).__name__}: {e}")

def load_plugins() -> list[BasePlugin]:
    """
    Instantiates and returns all discovered SonarQube metric plugin instances.

    Returns:
        list[BasePlugin]: List of instantiated Sonar metric plugin classes.
    """
    return [globals()[name]() for name in __all__]

# ✅ Load and expose plugin instances directly
plugins: list[BasePlugin] = load_plugins()

def get_sonar_metric_names() -> list[str]:
    """
    Returns all known metric names defined by the SonarQube plugins.

    Returns:
        list[str]: Metric identifier strings from each plugin.
    """
    return [plugin.name() for plugin in plugins]
