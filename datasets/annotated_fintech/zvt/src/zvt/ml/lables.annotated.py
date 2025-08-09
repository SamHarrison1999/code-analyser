# -*- coding: utf-8 -*-
# ✅ Best Practice: Use of Enum for defining a set of named constants improves code readability and maintainability.
from enum import Enum

# 🧠 ML Signal: Enum values can be used to categorize or label data, which is useful for ML classification tasks.

# ✅ Best Practice: Use of Enum for defining a set of named constants
class BehaviorCategory(Enum):
    # 🧠 ML Signal: Enum values can be used to categorize or label data, which is useful for ML classification tasks.
    # 上涨
    # ✅ Best Practice: Use of descriptive names for Enum members
    up = 1
    # 下跌
    # ✅ Best Practice: Use of __all__ to define public API of the module
    down = -1


class RelativePerformance(Enum):
    # 表现比90%好
    best = 0.9
    ordinary = 0.5
    poor = 0


# the __all__ is generated
__all__ = ["BehaviorCategory", "RelativePerformance"]