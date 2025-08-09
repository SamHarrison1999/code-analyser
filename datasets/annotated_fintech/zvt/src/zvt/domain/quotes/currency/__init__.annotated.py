# -*- coding: utf-8 -*-
# ✅ Best Practice: Explicitly importing specific items from a module improves readability and maintainability.


# ⚠️ SAST Risk (Medium): Using wildcard imports can lead to namespace pollution and make it unclear which names are present.
# ✅ Best Practice: Using a specific alias for imported items helps avoid naming conflicts and improves code clarity.
# 🧠 ML Signal: Modifying the __all__ list dynamically indicates a pattern of re-exporting symbols from imported modules.
# the __all__ is generated
__all__ = []

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule currency_1d_kdata
from .currency_1d_kdata import *
from .currency_1d_kdata import __all__ as _currency_1d_kdata_all

__all__ += _currency_1d_kdata_all