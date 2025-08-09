# -*- coding: utf-8 -*-#
# ✅ Best Practice: Use of wildcard imports can lead to namespace pollution and make it unclear which names are present in the namespace.


# 🧠 ML Signal: Importing __all__ from modules suggests a pattern of controlled public API exposure.
# the __all__ is generated
__all__ = []

# ✅ Best Practice: Use of wildcard imports can lead to namespace pollution and make it unclear which names are present in the namespace.
# __init__.py structure:
# common code of the package
# 🧠 ML Signal: Importing __all__ from modules suggests a pattern of controlled public API exposure.
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule holder
# ✅ Best Practice: Use of wildcard imports can lead to namespace pollution and make it unclear which names are present in the namespace.
from .holder import *
from .holder import __all__ as _holder_all

# 🧠 ML Signal: Importing __all__ from modules suggests a pattern of controlled public API exposure.

__all__ += _holder_all

# ✅ Best Practice: Use of wildcard imports can lead to namespace pollution and make it unclear which names are present in the namespace.
# import all from submodule trading
from .trading import *

# ✅ Best Practice: Use of wildcard imports can lead to namespace pollution and make it unclear which names are present in the namespace.
# 🧠 ML Signal: Importing __all__ from modules suggests a pattern of controlled public API exposure.
from .trading import __all__ as _trading_all

__all__ += _trading_all

# import all from submodule finance
from .finance import *
from .finance import __all__ as _finance_all

__all__ += _finance_all

# import all from submodule common
from .common import *
from .common import __all__ as _common_all

__all__ += _common_all

# import all from submodule dividend_financing
from .dividend_financing import *
from .dividend_financing import __all__ as _dividend_financing_all

__all__ += _dividend_financing_all

# import all from submodule meta
from .meta import *
from .meta import __all__ as _meta_all

__all__ += _meta_all
