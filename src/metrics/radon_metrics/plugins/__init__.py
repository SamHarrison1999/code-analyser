import pkgutil
import importlib
import inspect
import os

from .base import RadonMetricPlugin as BasePlugin

__all__ = []

# 🧠 Dynamically discover and import all valid plugin classes in the current directory
plugin_dir = os.path.dirname(__file__)
for _, module_name, _ in pkgutil.iter_modules([plugin_dir]):
    if module_name.startswith("_") or module_name == "base":
        continue  # 🚫 Skip private modules and base class
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                globals()[name] = obj
                __all__.append(name)
    except Exception as e:
        import logging
        logging.warning(f"[RadonPluginLoader] Skipped {module_name}: {type(e).__name__}: {e}")

# Alias for type-checking and consistency
RadonMetricPlugin = BasePlugin

def load_plugins() -> list[BasePlugin]:
    """
    Instantiates and returns all discovered Radon metric plugin instances.

    Returns:
        list[BasePlugin]: List of instantiated Radon metric plugin classes.
    """
    return [globals()[name]() for name in __all__]
