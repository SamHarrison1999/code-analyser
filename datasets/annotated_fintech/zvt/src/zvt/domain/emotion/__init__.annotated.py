# -*- coding: utf-8 -*-
# ✅ Best Practice: Importing specific modules or functions helps avoid namespace pollution.


# ✅ Best Practice: Explicitly managing __all__ helps control what is exposed when the module is imported.
# ⚠️ SAST Risk (Low): Using wildcard imports can lead to unexpected behavior due to namespace conflicts.
# 🧠 ML Signal: Wildcard imports may indicate a pattern of importing all available functions or classes from a module.
# ✅ Best Practice: Using __all__ to define public API of the module improves code maintainability and readability.
# the __all__ is generated
__all__ = []

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule emotion
from .emotion import *
from .emotion import __all__ as _emotion_all

__all__ += _emotion_all