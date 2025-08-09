# -*- coding: utf-8 -*-
# ✅ Best Practice: Explicitly importing all from a module can lead to namespace pollution; consider importing only necessary components.


# 🧠 ML Signal: Using __all__ to control what is exported from a module is a common pattern.
# the __all__ is generated
__all__ = []
# ✅ Best Practice: Using += to extend __all__ ensures that all exports from imported modules are included.

# ✅ Best Practice: Explicitly importing all from a module can lead to namespace pollution; consider importing only necessary components.
# 🧠 ML Signal: Using __all__ to control what is exported from a module is a common pattern.
# ✅ Best Practice: Using += to extend __all__ ensures that all exports from imported modules are included.
# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule monetary
from .monetary import *
from .monetary import __all__ as _monetary_all

__all__ += _monetary_all

# import all from submodule macro
from .macro import *
from .macro import __all__ as _macro_all

__all__ += _macro_all