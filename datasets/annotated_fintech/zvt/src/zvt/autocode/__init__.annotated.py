# -*- coding: utf-8 -*-#
# ✅ Best Practice: Importing specific items with '*' can lead to namespace pollution; consider importing only what is needed.
# the __all__ is generated
__all__ = []
# 🧠 ML Signal: Use of __all__ to control what is exported from a module.

# __init__.py structure:
# 🧠 ML Signal: Dynamically updating __all__ to include items from another module.
# common code of the package
# 🧠 ML Signal: Use of __all__ to control what is exported from a module.
# ✅ Best Practice: Importing specific items with '*' can lead to namespace pollution; consider importing only what is needed.
# 🧠 ML Signal: Dynamically updating __all__ to include items from another module.
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule generator
from .generator import *
from .generator import __all__ as _generator_all

__all__ += _generator_all

# import all from submodule templates
from .templates import *
from .templates import __all__ as _templates_all

__all__ += _templates_all
