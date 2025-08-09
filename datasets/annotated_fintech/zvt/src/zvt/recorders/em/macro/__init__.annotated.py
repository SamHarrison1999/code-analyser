# -*- coding: utf-8 -*-
# ✅ Best Practice: Explicitly importing specific items from a module improves readability and maintainability.


# ⚠️ SAST Risk (Medium): Using wildcard imports can lead to namespace pollution and make it unclear which names are present in the namespace.
# 🧠 ML Signal: Modifying the __all__ list indicates dynamic control over module exports, which can be a pattern for ML models to learn module interface management.
# the __all__ is generated
__all__ = []

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule em_treasury_yield_recorder
from .em_treasury_yield_recorder import *
from .em_treasury_yield_recorder import __all__ as _em_treasury_yield_recorder_all

__all__ += _em_treasury_yield_recorder_all
