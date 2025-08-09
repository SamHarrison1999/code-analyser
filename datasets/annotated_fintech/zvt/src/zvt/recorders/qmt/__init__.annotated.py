# -*- coding: utf-8 -*-#
# ✅ Best Practice: Explicitly importing specific items from modules can improve code readability and maintainability.


# ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace pollution and make it unclear which names are present.
# the __all__ is generated
__all__ = []
# ✅ Best Practice: Aggregating __all__ from submodules helps maintain a clear public API.

# __init__.py structure:
# ✅ Best Practice: Explicitly importing specific items from modules can improve code readability and maintainability.
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules
# ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace pollution and make it unclear which names are present.
# ✅ Best Practice: Aggregating __all__ from submodules helps maintain a clear public API.
# ✅ Best Practice: Explicitly importing specific items from modules can improve code readability and maintainability.

# import all from submodule quotes
from .quotes import *
from .quotes import __all__ as _quotes_all

__all__ += _quotes_all

# import all from submodule index
from .index import *
from .index import __all__ as _index_all

__all__ += _index_all

# import all from submodule meta
from .meta import *
from .meta import __all__ as _meta_all

__all__ += _meta_all
