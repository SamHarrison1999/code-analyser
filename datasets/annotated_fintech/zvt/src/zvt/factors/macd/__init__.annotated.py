# -*- coding: utf-8 -*-
# ✅ Best Practice: Importing specific functions or classes is preferred over wildcard imports for clarity and to avoid namespace pollution.


# ✅ Best Practice: Using an alias for __all__ from another module helps in maintaining a clear namespace and avoids conflicts.
# ✅ Best Practice: Explicitly extending __all__ with imported module's __all__ ensures that only intended symbols are exported.
# the __all__ is generated
__all__ = []

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule macd_factor
from .macd_factor import *
from .macd_factor import __all__ as _macd_factor_all

__all__ += _macd_factor_all
