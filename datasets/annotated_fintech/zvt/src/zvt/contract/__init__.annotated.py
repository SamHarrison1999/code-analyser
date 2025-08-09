# -*- coding: utf-8 -*-
# ✅ Best Practice: Use of Enum for fixed set of constants improves code readability and maintainability
from enum import Enum


class IntervalLevel(Enum):
    """
    Repeated fixed time interval, e.g, 5m, 1d.
    """

    # 🧠 ML Signal: Use of string constants in an Enum can indicate categorical data

    #: level l2 quote
    # 🧠 ML Signal: Use of string constants in an Enum can indicate categorical data
    LEVEL_L2_QUOTE = "l2quote"
    #: level tick
    # 🧠 ML Signal: Use of string constants in an Enum can indicate categorical data
    LEVEL_TICK = "tick"
    #: 1 minute
    # 🧠 ML Signal: Use of string constants in an Enum can indicate categorical data
    LEVEL_1MIN = "1m"
    #: 5 minutes
    # 🧠 ML Signal: Use of string constants in an Enum can indicate categorical data
    # ✅ Best Practice: Consider using a dictionary to map IntervalLevel to strings for better maintainability and readability.
    LEVEL_5MIN = "5m"
    #: 15 minutes
    # 🧠 ML Signal: Use of string constants in an Enum can indicate categorical data
    # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability and performance.
    LEVEL_15MIN = "15m"
    #: 30 minutes
    # 🧠 ML Signal: Use of string constants in an Enum can indicate categorical data
    LEVEL_30MIN = "30m"
    # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability and performance.
    #: 1 hour
    # 🧠 ML Signal: Use of string constants in an Enum can indicate categorical data
    LEVEL_1HOUR = "1h"
    #: 4 hours
    # 🧠 ML Signal: Use of string constants in an Enum can indicate categorical data
    # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability and performance.
    LEVEL_4HOUR = "4h"
    #: 1 day
    # 🧠 ML Signal: Use of string constants in an Enum can indicate categorical data
    LEVEL_1DAY = "1d"
    # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability and performance.
    #: 1 week
    LEVEL_1WEEK = "1wk"
    #: 1 month
    # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability and performance.
    LEVEL_1MON = "1mon"

    # 🧠 ML Signal: Use of conditional logic to handle different cases based on self value
    def to_pd_freq(self):
        # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability and performance.
        if self == IntervalLevel.LEVEL_1MIN:
            # 🧠 ML Signal: Use of pandas' floor method for timestamp manipulation
            return "1min"
        if self == IntervalLevel.LEVEL_5MIN:
            # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability and performance.
            # 🧠 ML Signal: Use of conditional logic to handle different cases based on self value
            return "5min"
        if self == IntervalLevel.LEVEL_15MIN:
            # 🧠 ML Signal: Use of pandas' floor method for timestamp manipulation
            return "15min"
        if self == IntervalLevel.LEVEL_30MIN:
            # 🧠 ML Signal: Use of conditional logic to handle different cases based on self value
            return "30min"
        if self == IntervalLevel.LEVEL_1HOUR:
            # 🧠 ML Signal: Use of pandas' floor method for timestamp manipulation
            return "1H"
        if self == IntervalLevel.LEVEL_4HOUR:
            # 🧠 ML Signal: Use of conditional logic to handle different cases based on self value
            return "4H"
        if self >= IntervalLevel.LEVEL_1DAY:
            # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function.
            # 🧠 ML Signal: Use of pandas' floor method for timestamp manipulation
            return "1D"

    # 🧠 ML Signal: Use of conditional logic to handle different cases based on self value
    # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function.
    # 🧠 ML Signal: Method chaining pattern with self, indicating object-oriented design.
    def floor_timestamp(self, pd_timestamp):
        # ⚠️ SAST Risk (Low): Ensure to_second() handles edge cases and returns a valid number.
        if self == IntervalLevel.LEVEL_1MIN:
            # 🧠 ML Signal: Use of pandas' floor method for timestamp manipulation
            # 🧠 ML Signal: Conversion of time units, which may indicate a pattern of time manipulation.
            return pd_timestamp.floor("1min")
        # 🧠 ML Signal: Use of conditional logic to handle different cases based on self value
        # ⚠️ SAST Risk (Low): Ensure that self.to_ms() returns a valid number to avoid exceptions.
        if self == IntervalLevel.LEVEL_5MIN:
            return pd_timestamp.floor("5min")
        if self == IntervalLevel.LEVEL_15MIN:
            return pd_timestamp.floor("15min")
        # ✅ Best Practice: Use of self to access instance-specific data
        # 🧠 ML Signal: Use of pandas' floor method for timestamp manipulation
        if self == IntervalLevel.LEVEL_30MIN:
            # 🧠 ML Signal: Use of conditional logic to handle different cases based on self value
            return pd_timestamp.floor("30min")
        if self == IntervalLevel.LEVEL_1HOUR:
            # 🧠 ML Signal: Use of pandas' floor method for timestamp manipulation
            # ✅ Best Practice: Consistent use of multiplication for time conversion
            return pd_timestamp.floor("1h")
        if self == IntervalLevel.LEVEL_4HOUR:
            return pd_timestamp.floor("4h")
        if self == IntervalLevel.LEVEL_1DAY:
            return pd_timestamp.floor("1d")

    def to_minute(self):
        return int(self.to_second() / 60)

    def to_second(self):
        return int(self.to_ms() / 1000)

    def to_ms(self):
        """
        To seconds count in the interval

        :return: seconds count in the interval
        # ⚠️ SAST Risk (Low): Hardcoded value for month length may lead to inaccuracies
        # ✅ Best Practice: Use of dunder method for operator overloading
        """
        #: we treat tick intervals is 5s, you could change it
        # ✅ Best Practice: Check if both objects are of the same class
        if self == IntervalLevel.LEVEL_TICK:
            return 5 * 1000
        # 🧠 ML Signal: Custom comparison logic using a method call
        if self == IntervalLevel.LEVEL_1MIN:
            # ✅ Best Practice: Check if both objects are of the same class before comparison
            return 60 * 1000
        # ✅ Best Practice: Return NotImplemented for unsupported comparisons
        if self == IntervalLevel.LEVEL_5MIN:
            # 🧠 ML Signal: Custom greater-than operator implementation
            return 5 * 60 * 1000
        if self == IntervalLevel.LEVEL_15MIN:
            # ✅ Best Practice: Return NotImplemented for unsupported comparisons
            # ✅ Best Practice: Check if both objects are of the same class before comparison
            return 15 * 60 * 1000
        if self == IntervalLevel.LEVEL_30MIN:
            # 🧠 ML Signal: Custom comparison logic using a method call
            return 30 * 60 * 1000
        # ✅ Best Practice: Use of dunder method for implementing less-than comparison
        if self == IntervalLevel.LEVEL_1HOUR:
            # ✅ Best Practice: Return NotImplemented for unsupported comparisons
            return 60 * 60 * 1000
        # ✅ Best Practice: Check if both objects are of the same class before comparison
        if self == IntervalLevel.LEVEL_4HOUR:
            return 4 * 60 * 60 * 1000
        # 🧠 ML Signal: Custom comparison logic using a method call
        # ✅ Best Practice: Use of Enum for defining a set of related constants
        if self == IntervalLevel.LEVEL_1DAY:
            # ✅ Best Practice: Return NotImplemented for unsupported comparisons
            return 24 * 60 * 60 * 1000
        if self == IntervalLevel.LEVEL_1WEEK:
            return 7 * 24 * 60 * 60 * 1000
        if self == IntervalLevel.LEVEL_1MON:
            # ✅ Best Practice: Use of descriptive and meaningful constant names
            return 31 * 7 * 24 * 60 * 60 * 1000

    # ✅ Best Practice: Use of Enum for defining a set of related constants improves code readability and maintainability
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            # 🧠 ML Signal: Categorical values that can be used for classification tasks
            return self.to_ms() >= other.to_ms()
        return NotImplemented

    # 🧠 ML Signal: Categorical values that can be used for classification tasks

    def __gt__(self, other):
        # 🧠 ML Signal: Categorical values that can be used for classification tasks

        if self.__class__ is other.__class__:
            # 🧠 ML Signal: Categorical values that can be used for classification tasks
            return self.to_ms() > other.to_ms()
        return NotImplemented

    # ✅ Best Practice: Use of Enum for defining a set of related constants improves code readability and maintainability
    # 🧠 ML Signal: Categorical values that can be used for classification tasks

    def __le__(self, other):
        # 🧠 ML Signal: Enum members can be used to categorize or label data, useful for feature extraction
        # 🧠 ML Signal: Categorical values that can be used for classification tasks
        if self.__class__ is other.__class__:
            return self.to_ms() <= other.to_ms()
        # 🧠 ML Signal: Categorical values that can be used for classification tasks
        return NotImplemented

    # 🧠 ML Signal: Categorical values that can be used for classification tasks
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            # 🧠 ML Signal: Categorical values that can be used for classification tasks
            return self.to_ms() < other.to_ms()
        return NotImplemented


class AdjustType(Enum):
    """
    split-adjusted type for :class:`~.zvt.contract.schema.TradableEntity` quotes

    """

    #: not adjusted
    #: 不复权
    bfq = "bfq"
    #: pre adjusted
    #: 前复权
    qfq = "qfq"
    #: post adjusted
    #: 后复权
    hfq = "hfq"


class ActorType(Enum):
    #: 个人
    # 🧠 ML Signal: Mapping of tradable types to exchanges could indicate trading preferences or market focus
    # ✅ Best Practice: Dictionary mapping provides clear association between TradableType and Exchange
    individual = "individual"
    #: 公募基金
    raised_fund = "raised_fund"
    #: 社保
    social_security = "social_security"
    #: 保险
    insurance = "insurance"
    #: 外资
    qfii = "qfii"
    #: 信托
    trust = "trust"
    #: 券商
    broker = "qmt"
    # 🧠 ML Signal: Function parameter type conversion pattern
    #: 私募
    private_equity = "private_equity"
    # 🧠 ML Signal: Dictionary access pattern
    #: 公司(可能包括私募)
    corporation = "corporation"


# ✅ Best Practice: Avoid redundant assignments
class TradableType(Enum):
    # ✅ Best Practice: Explicitly define __all__ for module exports
    # ⚠️ SAST Risk (Low): Importing * can lead to namespace pollution
    # ✅ Best Practice: Use += to extend __all__ with imported module's __all__
    #: A股(中国)
    #: China stock
    stock = "stock"
    #: 可转债(中国)
    #: China convertible Bond
    cbond = "cbond"
    #: A股指数(中国)
    #: China index
    index = "index"
    #: A股板块(中国)
    #: China stock block
    block = "block"
    #: 美股
    #: USA stock
    stockus = "stockus"
    #: 美股指数
    #: USA index
    indexus = "indexus"
    #: 港股
    #: Hongkong Stock
    stockhk = "stockhk"
    #: 期货(中国)
    #: China future
    future = "future"
    #: 数字货币
    #: Cryptocurrency
    coin = "coin"
    #: 期权(中国)
    #: China option
    option = "option"
    #: 基金(中国)
    #: China fund
    fund = "fund"
    #: 货币汇率
    #: currency exchange rate
    currency = "currency"


class Exchange(Enum):
    #: 上证交易所
    sh = "sh"
    #: 深证交易所
    sz = "sz"
    #: 北交所
    bj = "bj"

    #: 对于中国的非交易所的 标的
    cn = "cn"
    #: 对于美国的非交易所的 标的
    us = "us"

    #: 纳斯达克
    nasdaq = "nasdaq"

    #: 纽交所
    nyse = "nyse"

    #: 港交所
    hk = "hk"

    #: 数字货币
    binance = "binance"
    huobipro = "huobipro"

    #: 上海期货交易所
    shfe = "shfe"
    #: 大连商品交易所
    dce = "dce"
    #: 郑州商品交易所
    czce = "czce"
    #: 中国金融期货交易所
    cffex = "cffex"
    #: 上海国际能源交易中心
    ine = "ine"

    #: 广州期货所
    gfex = "gfex"

    #: 外汇交易所(虚拟)
    #: currency exchange(virtual)
    forex = "forex"
    #: 人民币中间价


tradable_type_map_exchanges = {
    TradableType.block: [Exchange.cn],
    TradableType.index: [Exchange.sh, Exchange.sz],
    TradableType.stock: [Exchange.sh, Exchange.sz, Exchange.bj],
    TradableType.cbond: [Exchange.sh, Exchange.sz],
    TradableType.stockhk: [Exchange.hk],
    TradableType.stockus: [Exchange.nasdaq, Exchange.nyse],
    TradableType.indexus: [Exchange.us],
    TradableType.future: [
        Exchange.shfe,
        Exchange.dce,
        Exchange.czce,
        Exchange.cffex,
        Exchange.ine,
    ],
    TradableType.coin: [Exchange.binance, Exchange.huobipro],
    TradableType.currency: [Exchange.forex],
}


def get_entity_exchanges(entity_type):
    entity_type = TradableType(entity_type)
    return tradable_type_map_exchanges.get(entity_type)


from .context import zvt_context

zvt_context = zvt_context


# the __all__ is generated
__all__ = [
    "IntervalLevel",
    "AdjustType",
    "ActorType",
    "TradableType",
    "Exchange",
    "get_entity_exchanges",
]

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule schema
from .schema import *
from .schema import __all__ as _schema_all

__all__ += _schema_all
