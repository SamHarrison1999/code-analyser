# -*- coding: utf-8 -*-
# ✅ Best Practice: Explicitly importing all from a module helps in understanding what is being imported.


# ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace collisions and make the code less readable.
# the __all__ is generated
__all__ = []
# 🧠 ML Signal: Pattern of importing and extending __all__ for module exports.

# __init__.py structure:
# ✅ Best Practice: Explicitly importing all from a module helps in understanding what is being imported.
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules
# ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace collisions and make the code less readable.

# import all from submodule eastmoney_rights_issue_detail_recorder
# 🧠 ML Signal: Pattern of importing and extending __all__ for module exports.
from .eastmoney_rights_issue_detail_recorder import *

# ✅ Best Practice: Explicitly importing all from a module helps in understanding what is being imported.
# ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace collisions and make the code less readable.
# 🧠 ML Signal: Pattern of importing and extending __all__ for module exports.
from .eastmoney_rights_issue_detail_recorder import (
    __all__ as _eastmoney_rights_issue_detail_recorder_all,
)

__all__ += _eastmoney_rights_issue_detail_recorder_all

# import all from submodule eastmoney_dividend_detail_recorder
from .eastmoney_dividend_detail_recorder import *
from .eastmoney_dividend_detail_recorder import (
    __all__ as _eastmoney_dividend_detail_recorder_all,
)

__all__ += _eastmoney_dividend_detail_recorder_all

# import all from submodule eastmoney_spo_detail_recorder
from .eastmoney_spo_detail_recorder import *
from .eastmoney_spo_detail_recorder import __all__ as _eastmoney_spo_detail_recorder_all

__all__ += _eastmoney_spo_detail_recorder_all

# import all from submodule eastmoney_dividend_financing_recorder
from .eastmoney_dividend_financing_recorder import *
from .eastmoney_dividend_financing_recorder import (
    __all__ as _eastmoney_dividend_financing_recorder_all,
)

__all__ += _eastmoney_dividend_financing_recorder_all
