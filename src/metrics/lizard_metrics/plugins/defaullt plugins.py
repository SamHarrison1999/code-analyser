from typing import Sequence
from pathlib import Path

from ...metric_types import MetricPlugin
from .base import BaseLizardPlugin
import inspect
import pkgutil
import importlib

from ..parser import load_lizard_output  # helper to run lizard & get output


def get_default_plugins() -> Sequence[MetricPlugin]:
    """
    Discovers and returns MetricPlugin definitions from all LizardPlugin subclasses.

    Returns:
        Sequence[MetricPlugin]: One plugin entry per metric plugin class.
    """
    plugins: list[MetricPlugin] = []

    # 🔍 Dynamically discover all plugin modules in this directory
    plugin_pkg = __name__.rsplit(".", 1)[0]
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        full_module = f"{plugin_pkg}.{module_name}"
        module = importlib.import_module(full_module)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseLizardPlugin) and obj is not BaseLizardPlugin:
                plugin_instance = obj()

                def make_extractor(plugin=obj) -> callable:
                    def extractor(file_path: Path):
                        lines = load_lizard_output(file_path)
                        value = plugin().extract(lines, str(file_path))
                        return [{
                            "name": plugin.name(),
                            "value": value,
                            "units": None,
                            "success": True,
                            "error": None
                        }]
                    return extractor

                plugins.append({
                    "name": obj.name(),
                    "type": "static_analysis",
                    "extractor": make_extractor(),
                    "domain": "code",
                    "language": "python",
                    "source": "lizard",
                    "version": "1.17.10",  # or dynamically resolve later
                    "format": "metrics",
                    "tool": "lizard",
                    "scope": "file",
                    "outputs": [obj.name()],
                })

    return plugins
