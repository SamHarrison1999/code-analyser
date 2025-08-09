# -*- coding: utf-8 -*-
# ✅ Best Practice: Importing specific attributes from a module can help avoid namespace pollution.
# the __all__ is generated
__all__ = []
# ⚠️ SAST Risk (Medium): Using wildcard imports can lead to namespace collisions and make the code harder to read.

# __init__.py structure:
# ✅ Best Practice: Explicitly managing __all__ to control what is exported from the module.
# common code of the package
# ✅ Best Practice: Importing specific attributes from a module can help avoid namespace pollution.
# ✅ Best Practice: Explicitly managing __all__ to control what is exported from the module.
# ⚠️ SAST Risk (Medium): Using wildcard imports can lead to namespace collisions and make the code harder to read.
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule lables
from .lables import *
from .lables import __all__ as _lables_all

__all__ += _lables_all

# import all from submodule ml
from .ml import *
from .ml import __all__ as _ml_all

__all__ += _ml_all
