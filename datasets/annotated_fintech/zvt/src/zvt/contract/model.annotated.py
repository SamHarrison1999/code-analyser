# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from datetime import datetime
# 🧠 ML Signal: Custom model class definition, useful for identifying model patterns

# ✅ Best Practice: Inherits from BaseModel, indicating use of a structured model approach
from pydantic import BaseModel, ConfigDict
# ✅ Best Practice: Class should inherit from object explicitly in Python 2.x for clarity, though not needed in Python 3.x

# 🧠 ML Signal: Configuration settings for the model, indicating customization

# ⚠️ SAST Risk (Low): Allowing inf/nan values might lead to unexpected behavior if not handled properly
# 🧠 ML Signal: Use of string type hints for id fields, common in database models
class CustomModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, allow_inf_nan=True)
# ✅ Best Practice: Use of __all__ to define public API of the module
# 🧠 ML Signal: Use of string type hints for entity_id fields, common in database models
# 🧠 ML Signal: Use of datetime type hints for timestamp fields, common in time-tracking models


class MixinModel(CustomModel):
    id: str
    entity_id: str
    timestamp: datetime


# the __all__ is generated
__all__ = ["CustomModel", "MixinModel"]