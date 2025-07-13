"""
Initialise Pyflakes metric plugins.

Dynamically loads all plugin implementations for Pyflakes diagnostics.
"""

import pkgutil
import importlib
import inspect
from typing import List
from metrics.pyflakes_metrics.plugins.base import PyflakesMetricPlugin
import metrics.pyflakes_metrics.plugins as plugin_module_root


def load_plugins() -> List[PyflakesMetricPlugin]:
    """
    Dynamically loads all subclasses of PyflakesPlugin.

    Returns:
        List[PyflakesMetricPlugin]: All discovered and instantiated plugins.
    """
    plugins = []
    for _, module_name, _ in pkgutil.iter_modules(plugin_module_root.__path__):
        module = importlib.import_module(f"{plugin_module_root.__name__}.{module_name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, PyflakesMetricPlugin)
                and obj is not PyflakesMetricPlugin
                and obj.__module__.startswith(plugin_module_root.__name__)
            ):
                plugins.append(obj())
    return plugins


__all__ = ["load_plugins", "PyflakesMetricPlugin"]
