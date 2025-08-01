# File: code_analyser/src/metrics/ast_metrics/plugins/__init__.py

# ✅ Best Practice: Use explicit imports to ensure introspection and safety
import pkgutil
import importlib
import inspect
import os

# ✅ Best Practice: Use a base class alias for clarity and consistency
from .base import ASTMetricPlugin as BasePlugin

# ✅ Best Practice: Collect available plugins and expose them publicly
__all__ = []
_discovered_plugins = {}

# ✅ Best Practice: Dynamically import all plugin modules in this directory
for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name == "base":
        # ✅ Best Practice: Skip base module to avoid unnecessary registration
        continue
    module = importlib.import_module(f"{__name__}.{module_name}")
    for name, obj in inspect.getmembers(module):
        # 🧠 ML Signal: Dynamically registered plugin classes suggest extensibility and modular architecture
        # ⚠️ SAST Risk: Ensure only expected subclasses are dynamically loaded
        if (
            inspect.isclass(obj)
            and issubclass(obj, BasePlugin)
            and obj is not BasePlugin
        ):
            globals()[name] = obj
            __all__.append(name)
            # ✅ Best Practice: Record plugin metadata for later use
            plugin_id = getattr(obj, "plugin_name", name)
            plugin_tags = getattr(obj, "plugin_tags", [])
            _discovered_plugins[plugin_id] = {
                "class": obj,
                "name": plugin_id,
                "tags": plugin_tags,
            }

# ✅ Best Practice: Expose the base plugin class under a consistent alias
ASTMetricPlugin = BasePlugin


# ✅ Best Practice: Load all available plugin instances
def load_plugins() -> list[BasePlugin]:
    return [entry["class"]() for entry in _discovered_plugins.values()]


# ✅ Best Practice: Load plugins filtered by tag (e.g., 'complexity', 'naming')
def load_plugins_by_tag(tag: str) -> list[BasePlugin]:
    return [
        entry["class"]()
        for entry in _discovered_plugins.values()
        if tag in entry["tags"]
    ]


# ✅ Best Practice: Load a specific plugin by name
def get_plugin_by_name(name: str) -> BasePlugin | None:
    entry = _discovered_plugins.get(name)
    return entry["class"]() if entry else None


# ✅ Best Practice: Expose metadata for UIs or logging
def list_plugins_metadata() -> list[dict]:
    return [
        {"name": entry["name"], "tags": entry["tags"]}
        for entry in _discovered_plugins.values()
    ]
