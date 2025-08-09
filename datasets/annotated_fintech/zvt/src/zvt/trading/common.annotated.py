# -*- coding: utf-8 -*-
# ✅ Best Practice: Use of Enum for defining a set of named values
from enum import Enum

# 🧠 ML Signal: Enum members for tracking execution states

class ExecutionStatus(Enum):
    # ✅ Best Practice: Use of __all__ to define public API of the module
    # 🧠 ML Signal: Enum members for tracking execution states
    init = "init"
    success = "success"
    failed = "failed"


# the __all__ is generated
__all__ = ["ExecutionStatus"]