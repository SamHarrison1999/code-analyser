# -*- coding: utf-8 -*-
# ✅ Best Practice: Explicit relative import for clarity within the package
# the __all__ is generated
__all__ = []
# ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace pollution and make it unclear which names are present
# ✅ Best Practice: Explicitly importing __all__ to manage namespace
# 🧠 ML Signal: Pattern of extending __all__ for module exports

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule informer
from .informer import *
from .informer import __all__ as _informer_all

__all__ += _informer_all
