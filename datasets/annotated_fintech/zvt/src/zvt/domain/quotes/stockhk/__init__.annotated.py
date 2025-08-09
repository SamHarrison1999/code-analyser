# -*- coding: utf-8 -*-
# ✅ Best Practice: Explicitly importing all from a module improves readability and maintainability.


# 🧠 ML Signal: Usage of wildcard imports can indicate a pattern of importing multiple functions or classes.
# the __all__ is generated
__all__ = []
# ✅ Best Practice: Aggregating __all__ from submodules helps in maintaining a clear public API.

# ✅ Best Practice: Explicitly importing all from a module improves readability and maintainability.
# 🧠 ML Signal: Usage of wildcard imports can indicate a pattern of importing multiple functions or classes.
# ✅ Best Practice: Aggregating __all__ from submodules helps in maintaining a clear public API.
# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule stockhk_1d_kdata
from .stockhk_1d_kdata import *
from .stockhk_1d_kdata import __all__ as _stockhk_1d_kdata_all

__all__ += _stockhk_1d_kdata_all

# import all from submodule stockhk_1d_hfq_kdata
from .stockhk_1d_hfq_kdata import *
from .stockhk_1d_hfq_kdata import __all__ as _stockhk_1d_hfq_kdata_all

__all__ += _stockhk_1d_hfq_kdata_all