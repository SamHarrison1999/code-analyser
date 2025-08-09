# -*- coding: utf-8 -*-
from datetime import datetime
from enum import Enum

# ✅ Best Practice: Use of Pydantic for data validation and settings management
from typing import Optional

# ✅ Best Practice: Use of Enum for defining a set of related constants improves code readability and maintainability.

from pydantic import BaseModel, Field

# ✅ Best Practice: Defining enum members with clear and descriptive names.

# ✅ Best Practice: Use of Enum for defining a set of related constants improves code readability and maintainability


class OrderByType(Enum):
    # ✅ Best Practice: Defining string values for Enum members enhances clarity and usability
    asc = "asc"
    desc = "desc"


class TimeUnit(Enum):
    # ✅ Best Practice: Inheriting from BaseModel suggests use of Pydantic for data validation and settings management
    year = "year"
    month = "month"
    # 🧠 ML Signal: Use of datetime for timestamps indicates time-based data processing
    day = "day"
    # ✅ Best Practice: Class should inherit from BaseModel to leverage pydantic's data validation features
    hour = "hour"
    # 🧠 ML Signal: Use of datetime for timestamps indicates time-based data processing
    minute = "minute"
    # ✅ Best Practice: Type hinting for class attributes improves code readability and maintainability
    second = "second"


# ✅ Best Practice: Inheriting from BaseModel suggests use of Pydantic for data validation and settings management.

# ✅ Best Practice: Type hinting for class attributes improves code readability and maintainability


# ✅ Best Practice: Use of Optional and default=None indicates that these fields are not mandatory.
class AbsoluteTimeRange(BaseModel):
    # ⚠️ SAST Risk (Low): Typo in 'RelativeTimeRage' could lead to runtime errors or misbehavior.
    # ✅ Best Practice: Use of __all__ to define public API of the module.
    start_timestamp: datetime
    end_timestamp: datetime


class RelativeTimeRage(BaseModel):
    interval: int
    time_unit: TimeUnit


class TimeRange(BaseModel):
    absolute_time_range: Optional[AbsoluteTimeRange] = Field(default=None)
    relative_time_range: Optional[RelativeTimeRage] = Field(default=None)


# the __all__ is generated
__all__ = [
    "OrderByType",
    "TimeUnit",
    "AbsoluteTimeRange",
    "RelativeTimeRage",
    "TimeRange",
]
