# -*- coding: utf-8 -*-

# ⚠️ SAST Risk (Medium): Using wildcard imports can lead to namespace pollution and make it unclear which names are present in the module.

# ✅ Best Practice: Explicitly importing __all__ to manage the public interface of the module.
# ✅ Best Practice: Updating __all__ to include imported module's public interface, ensuring controlled exposure of module contents.
# the __all__ is generated
__all__ = []

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule em_kdata_recorder
from .em_kdata_recorder import *
from .em_kdata_recorder import __all__ as _em_kdata_recorder_all

__all__ += _em_kdata_recorder_all
