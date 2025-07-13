"""
Pydocstyle Plugin Loader

Dynamically loads all subclasses of PydocstyleMetricPlugin from the plugins package.
"""

import pkgutil
import importlib
import inspect
from typing import List
from metrics.pydocstyle_metrics.plugins.base import PydocstyleMetricPlugin
import metrics.pydocstyle_metrics.plugins as plugin_module_root


def load_plugins() -> List[PydocstyleMetricPlugin]:
    """
    Discover and load all subclasses of PydocstylePlugin in the plugins package.

    Returns:
        List[PydocstyleMetricPlugin]: Instantiated plugin objects.
    """
    plugins = []
    for _, module_name, _ in pkgutil.iter_modules(plugin_module_root.__path__):
        module = importlib.import_module(f"{plugin_module_root.__name__}.{module_name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, PydocstyleMetricPlugin)
                and obj is not PydocstyleMetricPlugin
                and obj.__module__.startswith(plugin_module_root.__name__)
            ):
                plugins.append(obj())
    return plugins
