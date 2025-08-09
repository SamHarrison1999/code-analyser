# -*- coding: utf-8 -*-
# ✅ Best Practice: Group imports from the same module together
from datetime import datetime
# ✅ Best Practice: Class should be defined with proper indentation

from zvt.contract.model import CustomModel
# ✅ Best Practice: Type hinting improves code readability and maintainability

# ✅ Best Practice: Use of __all__ to define public API of the module
# ✅ Best Practice: Type hinting improves code readability and maintainability

class TimeMessage(CustomModel):
    # 时间
    timestamp: datetime
    # 信息
    message: str


# the __all__ is generated
__all__ = ["TimeMessage"]