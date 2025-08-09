# -*- coding: utf-8 -*-
# ✅ Best Practice: Use of Enum for predefined constant values improves code readability and maintainability
from enum import Enum

# ✅ Best Practice: Enum members provide a clear and concise way to define constant values


class StockPoolType(Enum):
    # ✅ Best Practice: Use of Enum for defining a set of named constants improves code readability and maintainability.
    system = "system"
    custom = "custom"
    # ✅ Best Practice: Defining specific string values for Enum members enhances clarity and usability.
    dynamic = "dynamic"


# ✅ Best Practice: Use of Enum for InsertMode improves code readability and maintainability


# ✅ Best Practice: Enum members are defined with clear and descriptive names
class DynamicPoolType(Enum):
    # ✅ Best Practice: Use of Enum for defining a set of related constants improves code readability and maintainability
    limit_up = "limit_up"
    limit_down = "limit_down"


# ✅ Best Practice: Defining class variables for Enum members improves code organization


# ✅ Best Practice: Use of Enum for defining a set of related constants improves code readability and maintainability.
class InsertMode(Enum):
    overwrite = "overwrite"
    append = "append"


# ✅ Best Practice: Defining __all__ to specify public symbols of the module improves code clarity and prevents unintended exports.


class TagType(Enum):
    #: A tag is a main tag due to its extensive capacity.
    main_tag = "main_tag"
    sub_tag = "sub_tag"
    hidden_tag = "hidden_tag"


class TagStatsQueryType(Enum):
    simple = "simple"
    details = "details"


# the __all__ is generated
__all__ = [
    "StockPoolType",
    "DynamicPoolType",
    "InsertMode",
    "TagType",
    "TagStatsQueryType",
]
