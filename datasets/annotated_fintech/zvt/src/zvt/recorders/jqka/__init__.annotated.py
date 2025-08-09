# -*- coding: utf-8 -*-
# ✅ Best Practice: Importing specific modules or functions helps avoid namespace pollution.


# ⚠️ SAST Risk (Medium): Using wildcard imports can lead to unexpected behavior and namespace conflicts.
# the __all__ is generated
__all__ = []
# ✅ Best Practice: Explicitly extending __all__ with imported module's __all__ for clarity.

# ✅ Best Practice: Importing specific modules or functions helps avoid namespace pollution.
# ⚠️ SAST Risk (Medium): Using wildcard imports can lead to unexpected behavior and namespace conflicts.
# ✅ Best Practice: Explicitly extending __all__ with imported module's __all__ for clarity.
# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule jqka_api
from .jqka_api import *
from .jqka_api import __all__ as _jqka_api_all

__all__ += _jqka_api_all

# import all from submodule emotion
from .emotion import *
from .emotion import __all__ as _emotion_all

__all__ += _emotion_all