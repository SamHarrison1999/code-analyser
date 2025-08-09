# -*- coding: utf-8 -*-
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

# 🧠 ML Signal: Use of Pydantic BaseModel for data validation and serialization
from zvt.contract import IntervalLevel
from zvt.trader import TradingSignalType

# 🧠 ML Signal: Use of type annotations for model attributes
from zvt.utils.time_utils import date_time_by_interval, current_date

# 🧠 ML Signal: Use of Optional for nullable fields


class FactorRequestModel(BaseModel):
    # 🧠 ML Signal: Use of default values in Pydantic models
    # ✅ Best Practice: Inheriting from BaseModel provides data validation and serialization.
    factor_name: str
    entity_ids: Optional[List[str]]
    # ⚠️ SAST Risk (Low): Potential timezone issues with datetime defaults
    # ✅ Best Practice: Using specific types for attributes improves code readability and type checking.
    data_provider: str = Field(default="em")
    # 🧠 ML Signal: Use of Field for default values and validation
    start_timestamp: datetime = Field(
        default=date_time_by_interval(current_date(), -365)
    )
    level: IntervalLevel = Field(default=IntervalLevel.LEVEL_1DAY)


# 🧠 ML Signal: Use of custom types in Pydantic models

# ✅ Best Practice: Providing default values for fields enhances usability and reduces errors.


class TradingSignalModel(BaseModel):
    entity_id: str
    # ✅ Best Practice: Using Optional for fields that can be None improves code clarity.
    happen_timestamp: datetime
    # 🧠 ML Signal: Use of Pydantic BaseModel for data validation and serialization
    due_timestamp: datetime
    trading_level: IntervalLevel = Field(default=IntervalLevel.LEVEL_1DAY)
    # 🧠 ML Signal: Use of Optional type hint indicating nullable field
    # 🧠 ML Signal: Use of type hint for data validation and clarity
    # ✅ Best Practice: Use of __all__ to define public API of the module
    trading_signal_type: TradingSignalType
    position_pct: Optional[float] = Field(default=0.2)
    order_amount: Optional[float] = Field(default=None)
    order_money: Optional[float] = Field(default=None)


class FactorResultModel(BaseModel):
    entity_ids: Optional[List[str]]
    tag_reason: str


# the __all__ is generated
__all__ = ["FactorRequestModel", "TradingSignalModel", "FactorResultModel"]
