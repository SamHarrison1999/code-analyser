# -*- coding: utf-8 -*-
from datetime import datetime
from typing import List, Optional
from typing import Union

# ✅ Best Practice: Grouping imports from the same module in a single line improves readability.
from pydantic import BaseModel, Field
from pydantic import field_validator

from zvt.common.query_models import TimeRange, OrderByType
from zvt.contract import IntervalLevel, AdjustType
from zvt.contract.model import MixinModel, CustomModel
from zvt.tag.tag_utils import get_stock_pool_names

# ✅ Best Practice: Grouping imports from the same module in a single line improves readability.
from zvt.trader import TradingSignalType

# 🧠 ML Signal: Use of Pydantic BaseModel for data validation and serialization
from zvt.trading.common import ExecutionStatus
from zvt.utils.time_utils import date_time_by_interval, current_date

# 🧠 ML Signal: List of strings for entity IDs, common pattern for handling multiple identifiers
from zvt.utils.time_utils import tomorrow_date, to_pd_timestamp

# ✅ Best Practice: Default value for data_provider improves usability and reduces errors


class KdataRequestModel(BaseModel):
    # ⚠️ SAST Risk (Low): Potential timezone issues with datetime defaults
    entity_ids: List[str]
    # ✅ Best Practice: Class should inherit from BaseModel for data validation and parsing
    # ✅ Best Practice: Use of Field with default value for start_timestamp ensures consistent initialization
    data_provider: str = Field(default="qmt")
    start_timestamp: datetime = Field(
        default=date_time_by_interval(current_date(), -500)
    )
    # ✅ Best Practice: Use of Optional for end_timestamp indicates that it can be None
    # 🧠 ML Signal: Use of string type for entity_id, indicating unique identifiers
    end_timestamp: Optional[datetime] = Field(default=None)
    level: IntervalLevel = Field(default=IntervalLevel.LEVEL_1DAY)
    # ✅ Best Practice: Default value for level provides a sensible default for interval level
    # 🧠 ML Signal: Use of string type for code, indicating stock or asset codes
    adjust_type: AdjustType = Field(default=AdjustType.qfq)


# 🧠 ML Signal: Use of Pydantic BaseModel for data validation and model definition
# ✅ Best Practice: Default value for adjust_type provides a sensible default for adjustment type
# 🧠 ML Signal: Use of string type for name, indicating descriptive labels


class KdataModel(BaseModel):
    # 🧠 ML Signal: List of strings indicating entity identifiers
    # ✅ Best Practice: Default values for fields improve usability and reduce errors
    entity_id: str
    code: str
    # 🧠 ML Signal: Default value for data_provider indicating a common or preferred provider
    # ⚠️ SAST Risk (Low): Type hinting without initialization can lead to runtime errors if not handled properly
    name: str
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
    level: IntervalLevel = Field(default=IntervalLevel.LEVEL_1DAY)
    # 🧠 ML Signal: Default value for days_count indicating a common or preferred time range
    datas: List


# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.


# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
# ✅ Best Practice: Inheriting from BaseModel provides data validation and serialization features.
class TSRequestModel(BaseModel):
    entity_ids: List[str]
    # ⚠️ SAST Risk (Low): The type for 'datas' is not fully specified, which could lead to unexpected behavior if not handled properly.
    # ✅ Best Practice: Using type annotations for class attributes improves code readability and type checking.
    data_provider: str = Field(default="qmt")
    days_count: int = Field(default=5)


class TSModel(BaseModel):
    entity_id: str
    code: str
    name: str
    # ✅ Best Practice: Using Optional for fields that can be None improves code clarity and type safety.
    # ✅ Best Practice: Inheriting from a custom model class suggests a structured approach to model definition
    datas: List


# ✅ Best Practice: Using Optional for fields that can be None improves code clarity and type safety.
# ✅ Best Practice: Using Optional type hints improves code readability and indicates that the field can be None


class QuoteStatsModel(BaseModel):
    # ✅ Best Practice: Using Optional type hints improves code readability and indicates that the field can be None
    #: UNIX时间戳
    # ✅ Best Practice: Using List type hint provides clarity on the expected data structure
    # ✅ Best Practice: Use of Optional and default value for main_tags improves code readability and maintainability
    time: int
    #: 涨停数
    limit_up_count: int
    # ✅ Best Practice: Use of field_validator decorator for input validation
    # ✅ Best Practice: Use of class method decorator for methods that don't modify class state
    #: 跌停数
    limit_down_count: int
    # ✅ Best Practice: Clear and descriptive variable naming
    #: 上涨数
    # 🧠 ML Signal: Pattern of checking membership in a list or collection
    up_count: int
    # ✅ Best Practice: Class should inherit from a base class to ensure consistent behavior and structure
    #: 下跌数
    # ⚠️ SAST Risk (Low): Potential information disclosure through error messages
    down_count: int
    # ✅ Best Practice: Type hinting for attributes improves code readability and maintainability
    #: 涨幅
    # 🧠 ML Signal: Use of Optional fields indicates handling of missing or default values
    change_pct: float
    # ✅ Best Practice: Use of Optional and default values improves code robustness and readability
    # ✅ Best Practice: Type hinting for attributes improves code readability and maintainability
    #: 成交额
    turnover: float
    # 🧠 ML Signal: Use of Optional fields indicates handling of missing or default values
    #: 昨日成交额
    pre_turnover: Optional[float] = Field(default=None)
    # 🧠 ML Signal: Use of Optional fields indicates handling of missing or default values
    #: 同比
    turnover_change: Optional[float] = Field(default=None)


# 🧠 ML Signal: Use of Optional fields indicates handling of missing or default values
# 🧠 ML Signal: Definition of a data model class, useful for understanding data structures in ML models


# ✅ Best Practice: Setting a default value for limit ensures consistent behavior
# 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features
class QueryStockQuoteSettingModel(CustomModel):
    stock_pool_name: Optional[str] = Field(default=None)
    # 🧠 ML Signal: Use of Optional fields indicates handling of missing or default values
    # 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features
    main_tags: Optional[List[str]] = Field(default=None)


# 🧠 ML Signal: Use of Optional fields indicates handling of missing or default values
# 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features


class BuildQueryStockQuoteSettingModel(CustomModel):
    # 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features
    stock_pool_name: str
    main_tags: Optional[List[str]] = Field(default=None)
    # 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features

    @field_validator("stock_pool_name")
    # 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features
    @classmethod
    def stock_pool_name_existed(cls, v: str) -> str:
        # 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features
        if v not in get_stock_pool_names():
            raise ValueError(f"Invalid stock_pool_name: {v}")
        # 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features
        return v


# 🧠 ML Signal: Use of Optional type for attributes, indicates nullable fields in data models


# ✅ Best Practice: Class definition should inherit from a base class for consistency and potential reuse
class QueryTagQuoteModel(CustomModel):
    # 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features
    stock_pool_name: str
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability
    main_tags: List[str]


# 🧠 ML Signal: Use of Optional type for attributes, indicates nullable fields in data models

# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability


# 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features
class QueryStockQuoteModel(CustomModel):
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability

    # 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features
    main_tag: Optional[str] = Field(default=None)
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability
    entity_ids: Optional[List[str]] = Field(default=None)
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
    # 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features
    stock_pool_name: Optional[str] = Field(default=None)
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability
    # the amount is not huge, just ignore now
    # 🧠 ML Signal: Use of type annotations for attributes, helps in understanding data types for ML features
    limit: int = Field(default=100)
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability
    order_by_type: Optional[OrderByType] = Field(default=OrderByType.desc)
    # 🧠 ML Signal: Use of Optional type for attributes, indicates nullable fields in data models
    order_by_field: Optional[str] = Field(default="change_pct")


# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability

# 🧠 ML Signal: Use of Union type for attributes, indicates multiple possible types for a field


# ✅ Best Practice: Use of type annotations for complex types like lists enhances code clarity.
class StockQuoteModel(MixinModel):
    # 🧠 ML Signal: Use of stock identifiers and trading details can indicate a financial trading application
    # 🧠 ML Signal: Use of Union type for attributes, indicates multiple possible types for a field
    #: 代码
    code: str
    # 🧠 ML Signal: Use of stock identifiers and trading details can indicate a financial trading application
    #: 名字
    name: str
    # 🧠 ML Signal: Use of stock identifiers and trading details can indicate a financial trading application

    #: UNIX时间戳
    # 🧠 ML Signal: Use of trading_date can indicate time-based operations or predictions
    time: int
    #: 最新价
    # 🧠 ML Signal: Use of expected_open_pct can indicate predictive modeling for stock prices
    price: float
    # 涨跌幅
    # 🧠 ML Signal: Optional buy_price suggests conditional trading strategies
    change_pct: float
    # 成交金额
    # 🧠 ML Signal: Optional sell_price suggests conditional trading strategies
    turnover: float
    # 换手率
    # 🧠 ML Signal: Use of trading_reason can indicate decision-making processes in trading
    turnover_rate: float
    #: 是否涨停
    # 🧠 ML Signal: Use of trading_signal_type can indicate categorization of trading strategies
    is_limit_up: bool
    #: 封涨停金额
    # ✅ Best Practice: Use of field_validator to ensure data integrity for trading_date
    # 🧠 ML Signal: Default status indicates initial state management in trading execution
    # ✅ Best Practice: Use of default values for fields improves model robustness
    limit_up_amount: Optional[float] = Field(default=None)
    #: 是否跌停
    is_limit_down: bool
    # ✅ Best Practice: Use of type hints for function parameters and return type
    # 🧠 ML Signal: Optional review suggests post-trade analysis or feedback mechanisms
    #: 封跌停金额
    limit_down_amount: Optional[float] = Field(default=None)
    # 🧠 ML Signal: Use of a utility function to convert string to timestamp
    #: 5挡卖单金额
    # ⚠️ SAST Risk (Low): Potential timezone issues if to_pd_timestamp does not handle timezones
    ask_amount: float
    #: 5挡买单金额
    # ⚠️ SAST Risk (Low): Error message may expose sensitive information if not handled properly
    # 🧠 ML Signal: Inheritance from BaseModel suggests usage of a data validation library like Pydantic.
    bid_amount: float
    # ✅ Best Practice: Returning the validated input value
    # ✅ Best Practice: Use of __all__ to define public API of the module.
    #: 流通市值
    float_cap: float
    #: 总市值
    total_cap: float

    main_tag: Optional[str] = Field(default=None)
    sub_tag: Union[str, None] = Field(default=None)
    hidden_tags: Union[List[str], None] = Field(default=None)


class TagQuoteStatsModel(CustomModel):
    main_tag: str
    #: 涨停数
    limit_up_count: int
    #: 跌停数
    limit_down_count: int
    #: 上涨数
    up_count: int
    #: 下跌数
    down_count: int
    #: 涨幅
    change_pct: float
    #: 成交额
    turnover: float


class StockQuoteStatsModel(CustomModel):
    #: 涨停数
    limit_up_count: int
    #: 跌停数
    limit_down_count: int
    #: 上涨数
    up_count: int
    #: 下跌数
    down_count: int
    #: 涨幅
    change_pct: float
    #: 成交额
    turnover: float

    quotes: List[StockQuoteModel]


class TradingPlanModel(MixinModel):
    stock_id: str
    stock_code: str
    stock_name: str
    # 执行交易日
    trading_date: datetime
    # 预期开盘涨跌幅
    expected_open_pct: float
    buy_price: Optional[float]
    sell_price: Optional[float]
    # 操作理由
    trading_reason: str
    # 交易信号
    trading_signal_type: TradingSignalType
    # 执行状态
    status: ExecutionStatus = Field(default=ExecutionStatus.init)
    # 复盘
    review: Optional[str]


class BuildTradingPlanModel(BaseModel):
    stock_id: str
    # 执行交易日
    trading_date: datetime
    # 预期开盘涨跌幅
    expected_open_pct: float
    buy_price: Optional[float]
    sell_price: Optional[float]
    # 操作理由
    trading_reason: str
    # 交易信号
    trading_signal_type: TradingSignalType

    @field_validator("trading_date")
    @classmethod
    def trading_date_must_be_future(cls, v: str) -> str:
        if to_pd_timestamp(v) < tomorrow_date():
            raise ValueError(f"trading_date: {v} must set to future trading date")
        return v


class QueryTradingPlanModel(BaseModel):
    time_range: TimeRange


# the __all__ is generated
__all__ = [
    "QueryTagQuoteModel",
    "QueryStockQuoteSettingModel",
    "BuildQueryStockQuoteSettingModel",
    "QueryStockQuoteModel",
    "StockQuoteModel",
    "StockQuoteStatsModel",
    "TradingPlanModel",
    "BuildTradingPlanModel",
    "QueryTradingPlanModel",
]
