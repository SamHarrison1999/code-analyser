# -*- coding: utf-8 -*-
# ✅ Best Practice: Explicitly importing specific items from a module improves readability and maintainability.


# ⚠️ SAST Risk (Medium): Using wildcard imports can lead to namespace collisions and make it unclear which names are present.
# ✅ Best Practice: Using a separate variable for imported __all__ helps avoid overwriting the local __all__ unintentionally.
# 🧠 ML Signal: Modifying __all__ dynamically can indicate a pattern of re-exporting symbols from submodules.
# the __all__ is generated
__all__ = []

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule indexus_1d_kdata
from .indexus_1d_kdata import *
from .indexus_1d_kdata import __all__ as _indexus_1d_kdata_all

__all__ += _indexus_1d_kdata_all