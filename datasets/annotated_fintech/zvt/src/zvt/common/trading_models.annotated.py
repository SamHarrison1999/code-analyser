# -*- coding: utf-8 -*-
from enum import Enum
from typing import List, Optional
# ✅ Best Practice: Use of Enum for defining a set of named values

from pydantic import BaseModel, Field
# ✅ Best Practice: Clear and descriptive naming for enum members

# ✅ Best Practice: Inheriting from BaseModel provides data validation and serialization.

# ✅ Best Practice: Clear and descriptive naming for enum members
class PositionType(Enum):
    # 🧠 ML Signal: Usage of List[str] indicates a collection of string identifiers.
    # 按整体仓位算
    normal = "normal"
    # 🧠 ML Signal: Default values for enums can indicate common or expected usage patterns.

    # 不管整体仓位多少
    # 🧠 ML Signal: Optional fields with default None can indicate optional parameters in usage.
    # ✅ Best Practice: Inheriting from BaseModel provides data validation and serialization features.
    # 按现金算
    cash = "cash"
# 🧠 ML Signal: Optional fields with default None can indicate optional parameters in usage.
# 🧠 ML Signal: Usage of List[str] indicates a pattern of handling multiple string identifiers.


# 🧠 ML Signal: Optional fields with default None can indicate optional parameters in usage.
# ✅ Best Practice: Use of Optional and List from typing for type hinting improves code readability and maintainability.
# 🧠 ML Signal: Optional[List[float]] suggests handling of optional numerical data, which can be a pattern in financial models.
class BuyParameter(BaseModel):
    # ✅ Best Practice: Using Field with default=None is a clear way to define optional fields with default values.
    entity_ids: List[str]
    # ✅ Best Practice: Use of Optional and List from typing for type hinting improves code readability and maintainability.
    # ✅ Best Practice: Defining __all__ helps in controlling what is exported when the module is imported using 'from module import *'.
    position_type: PositionType = Field(default=PositionType.normal)
    position_pct: Optional[float] = Field(default=None)
    weights: Optional[List[float]] = Field(default=None)
    money_to_use: Optional[float] = Field(default=None)


class SellParameter(BaseModel):
    entity_ids: List[str]
    sell_pcts: Optional[List[float]] = Field(default=None)


class TradingResult(BaseModel):
    success_entity_ids: Optional[List[str]] = Field(default=None)
    failed_entity_ids: Optional[List[str]] = Field(default=None)


# the __all__ is generated
__all__ = ["PositionType", "BuyParameter", "SellParameter", "TradingResult"]