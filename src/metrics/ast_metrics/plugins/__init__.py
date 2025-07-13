import pkgutil
import importlib
import inspect
import os

from .base import ASTMetricPlugin as BasePlugin

__all__ = []

for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name == "base":
        continue
    module = importlib.import_module(f"{__name__}.{module_name}")
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BasePlugin) and obj is not BasePlugin:
            globals()[name] = obj
            __all__.append(name)

ASTMetricPlugin = BasePlugin

def load_plugins() -> list[BasePlugin]:
    return [globals()[name]() for name in __all__]

