# -*- coding: utf-8 -*-
# ✅ Best Practice: Explicitly importing specific items from a module improves readability and maintainability.


# ⚠️ SAST Risk (Medium): Using wildcard imports can lead to namespace pollution and make it unclear which names are present in the namespace.
# ✅ Best Practice: Using a specific alias for imported items can prevent name conflicts and improve code clarity.
# 🧠 ML Signal: Modifying the __all__ variable indicates dynamic control over what is exposed by the module.
# the __all__ is generated
__all__ = []

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule em_stock_news_recorder
from .em_stock_news_recorder import *
from .em_stock_news_recorder import __all__ as _em_stock_news_recorder_all

__all__ += _em_stock_news_recorder_all