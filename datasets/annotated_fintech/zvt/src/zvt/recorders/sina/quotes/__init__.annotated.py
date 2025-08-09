# -*- coding: utf-8 -*-#
# ✅ Best Practice: Initialize __all__ to control what is exported when using 'from module import *'


# the __all__ is generated
# ⚠️ SAST Risk (Medium): Using 'import *' can lead to namespace pollution and make it unclear which names are present in the namespace
__all__ = []
# 🧠 ML Signal: Usage of 'import *' indicates a pattern of importing all module contents, which can be a feature for ML models

# ✅ Best Practice: Importing __all__ to extend the current module's __all__ list ensures explicit control over exports
# ✅ Best Practice: Extending __all__ with specific module exports maintains clarity on what is being exported
# ⚠️ SAST Risk (Medium): Using 'import *' can lead to namespace pollution and make it unclear which names are present in the namespace
# 🧠 ML Signal: Usage of 'import *' indicates a pattern of importing all module contents, which can be a feature for ML models
# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule sina_etf_kdata_recorder
from .sina_etf_kdata_recorder import *
from .sina_etf_kdata_recorder import __all__ as _sina_etf_kdata_recorder_all

__all__ += _sina_etf_kdata_recorder_all

# import all from submodule sina_index_kdata_recorder
from .sina_index_kdata_recorder import *
from .sina_index_kdata_recorder import __all__ as _sina_index_kdata_recorder_all

__all__ += _sina_index_kdata_recorder_all
