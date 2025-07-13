import pkgutil
import importlib
import inspect
import os
import logging

from .base import Flake8MetricPlugin as BasePlugin

logger = logging.getLogger(__name__)
__all__ = []

# 🔍 Discover all modules in this directory except `base`
for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name == "base":
        continue
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BasePlugin) and obj is not BasePlugin:
                globals()[name] = obj
                __all__.append(name)
                logger.debug(f"✅ Loaded Flake8 plugin: {name} from {module_name}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load Flake8 plugin module '{module_name}': {type(e).__name__}: {e}")

# Alias for base class
Flake8MetricPlugin = BasePlugin

def load_plugins() -> list[BasePlugin]:
    """Instantiate and return all discovered Flake8 metric plugin classes."""
    return [globals()[name]() for name in __all__]
