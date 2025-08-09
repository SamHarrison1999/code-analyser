# -*- coding: utf-8 -*-
# ✅ Best Practice: Importing specific items with '*' can lead to namespace pollution; consider importing only necessary items.


# ✅ Best Practice: Using an alias for __all__ helps avoid conflicts and improves readability.
# the __all__ is generated
__all__ = []
# 🧠 ML Signal: Dynamic modification of __all__ can indicate module composition patterns.

# __init__.py structure:
# ✅ Best Practice: Importing specific items with '*' can lead to namespace pollution; consider importing only necessary items.
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules
# ✅ Best Practice: Using an alias for __all__ helps avoid conflicts and improves readability.
# 🧠 ML Signal: Dynamic modification of __all__ can indicate module composition patterns.
# ✅ Best Practice: Importing specific items with '*' can lead to namespace pollution; consider importing only necessary items.

# import all from submodule wb_economy_recorder
from .wb_economy_recorder import *
from .wb_economy_recorder import __all__ as _wb_economy_recorder_all

__all__ += _wb_economy_recorder_all

# import all from submodule wb_country_recorder
from .wb_country_recorder import *
from .wb_country_recorder import __all__ as _wb_country_recorder_all

__all__ += _wb_country_recorder_all

# import all from submodule wb_api
from .wb_api import *
from .wb_api import __all__ as _wb_api_all

__all__ += _wb_api_all