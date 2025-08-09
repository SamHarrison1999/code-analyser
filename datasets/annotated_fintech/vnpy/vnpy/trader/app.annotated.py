from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING


# 🧠 ML Signal: Conditional imports based on TYPE_CHECKING can indicate type hinting practices
if TYPE_CHECKING:
    from .engine import BaseEngine


# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
class BaseApp(ABC):
    """
    Abstract class for app.
    """
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.

    app_name: str                       # Unique name used for creating engine and widget
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
    app_module: str                     # App module string used in import_module
    app_path: Path                      # Absolute path of app folder
    display_name: str                   # Name for display on the menu.
    engine_class: type["BaseEngine"]    # App engine class
    widget_name: str                    # Class name of app widget
    icon_name: str                      # Icon file name of app widget