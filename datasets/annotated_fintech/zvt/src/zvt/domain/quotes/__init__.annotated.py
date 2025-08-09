# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy import String, Column, Float, Integer, JSON

# 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema
from zvt.contract import Mixin

# 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema

class KdataCommon(Mixin):
    # 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema
    provider = Column(String(length=32))
    code = Column(String(length=32))
    # 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema
    name = Column(String(length=32))
    # Enum constraint is not extendable
    # 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema
    # level = Column(Enum(IntervalLevel, values_callable=enum_value))
    level = Column(String(length=32))
    # 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema

    # 开盘价
    # 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema
    # 🧠 ML Signal: Class definition with multiple attributes indicates a data model pattern
    open = Column(Float)
    # 收盘价
    # 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping
    # 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema
    close = Column(Float)
    # 最高价
    # 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping
    # 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema
    high = Column(Float)
    # 最低价
    # 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping
    # 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema
    low = Column(Float)
    # 成交量
    # 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping
    # 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema
    volume = Column(Float)
    # 成交金额
    # 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping
    # 🧠 ML Signal: Use of SQLAlchemy's Column to define database schema
    turnover = Column(Float)
    # 涨跌幅
    # 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping
    change_pct = Column(Float)
    # ✅ Best Practice: Use of inheritance to extend functionality from a base class
    # 换手率
    # 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping
    turnover_rate = Column(Float)
# ✅ Best Practice: Use of 'pass' to indicate an intentionally empty class definition
# ✅ Best Practice: Use of inheritance to extend functionality from a base class

# 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping

# ✅ Best Practice: Use of inheritance to extend functionality from a base class
# ✅ Best Practice: Use of 'pass' to indicate an intentionally empty class
class TickCommon(Mixin):
    # 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping
    #: UNIX时间戳
    # ✅ Best Practice: Class should have a docstring to describe its purpose and usage
    # ✅ Best Practice: Use of 'pass' to indicate an intentionally empty class definition
    time = Column(Integer)
    # 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping
    #: 开盘价
    # ✅ Best Practice: Class attributes should have comments or docstrings explaining their purpose
    open = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping
    # ✅ Best Practice: Use of inheritance to extend functionality from a base class
    #: 收盘价/当前价格
    # ✅ Best Practice: Class attributes should have comments or docstrings explaining their purpose
    close = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column for ORM mapping
    # ✅ Best Practice: Use of inheritance to extend functionality from a base class
    #: 最高价
    high = Column(Float)
    # ✅ Best Practice: Use of inheritance to extend functionality from a base class
    #: 最低价
    low = Column(Float)
    # ✅ Best Practice: Class should inherit from a base class to promote code reuse and maintainability
    #: 成交量
    volume = Column(Float)
    # ✅ Best Practice: Class attributes are defined using SQLAlchemy's Column, which is a common pattern for ORM models
    #: 成交金额
    turnover = Column(Float)
    # ✅ Best Practice: Class attributes are defined using SQLAlchemy's Column, which is a common pattern for ORM models
    #: 委卖价
    ask_price = Column(Float)
    # ✅ Best Practice: Class attributes are defined using SQLAlchemy's Column, which is a common pattern for ORM models
    #: 委买价
    bid_price = Column(Float)
    # 🧠 ML Signal: Use of __all__ to define public API of the module
    #: 委卖量
    ask_vol = Column(JSON)
    #: 委买量
    bid_vol = Column(JSON)
    #: 成交笔数
    transaction_num = Column(Integer)


class BlockKdataCommon(KdataCommon):
    pass


class IndexKdataCommon(KdataCommon):
    pass
# ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace pollution


# 🧠 ML Signal: Use of __all__ to manage namespace exports
class IndexusKdataCommon(KdataCommon):
    pass
# 🧠 ML Signal: Extending __all__ with imported module's __all__


# ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace pollution
class EtfKdataCommon(KdataCommon):
    turnover_rate = Column(Float)
    # 🧠 ML Signal: Use of __all__ to manage namespace exports

    # ETF 累计净值（货币 ETF 为七日年化)
    # 🧠 ML Signal: Extending __all__ with imported module's __all__
    cumulative_net_value = Column(Float)

# ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace pollution

class StockKdataCommon(KdataCommon):
    # 🧠 ML Signal: Use of __all__ to manage namespace exports
    pass

# 🧠 ML Signal: Extending __all__ with imported module's __all__

class StockusKdataCommon(KdataCommon):
    # ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace pollution
    pass

# 🧠 ML Signal: Use of __all__ to manage namespace exports

class StockhkKdataCommon(KdataCommon):
    # 🧠 ML Signal: Extending __all__ with imported module's __all__
    pass

# ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace pollution

# future common kdata
# 🧠 ML Signal: Use of __all__ to manage namespace exports
class FutureKdataCommon(KdataCommon):
    #: 持仓量
    # ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace pollution
    # 🧠 ML Signal: Extending __all__ with imported module's __all__
    # 🧠 ML Signal: Use of __all__ to manage namespace exports
    interest = Column(Float)
    #: 结算价
    settlement = Column(Float)
    #: 涨跌幅(按收盘价)
    # change_pct = Column(Float)
    #: 涨跌幅(按结算价)
    change_pct1 = Column(Float)


class CurrencyKdataCommon(KdataCommon):
    #: 持仓量
    interest = Column(Float)
    #: 结算价
    settlement = Column(Float)
    #: 涨跌幅(按收盘价)
    # change_pct = Column(Float)
    #: 涨跌幅(按结算价)
    change_pct1 = Column(Float)


# the __all__ is generated
__all__ = [
    "KdataCommon",
    "TickCommon",
    "BlockKdataCommon",
    "IndexKdataCommon",
    "IndexusKdataCommon",
    "EtfKdataCommon",
    "StockKdataCommon",
    "StockusKdataCommon",
    "StockhkKdataCommon",
    "FutureKdataCommon",
    "CurrencyKdataCommon",
]

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule trade_day
from .trade_day import *
from .trade_day import __all__ as _trade_day_all

__all__ += _trade_day_all

# import all from submodule indexus
from .indexus import *
from .indexus import __all__ as _indexus_all

__all__ += _indexus_all

# import all from submodule stockhk
from .stockhk import *
from .stockhk import __all__ as _stockhk_all

__all__ += _stockhk_all

# import all from submodule stockus
from .stockus import *
from .stockus import __all__ as _stockus_all

__all__ += _stockus_all

# import all from submodule index
from .index import *
from .index import __all__ as _index_all

__all__ += _index_all

# import all from submodule etf
from .etf import *
from .etf import __all__ as _etf_all

__all__ += _etf_all

# import all from submodule stock
from .stock import *
from .stock import __all__ as _stock_all

__all__ += _stock_all

# import all from submodule currency
from .currency import *
from .currency import __all__ as _currency_all

__all__ += _currency_all

# import all from submodule future
from .future import *
from .future import __all__ as _future_all

__all__ += _future_all

# import all from submodule block
from .block import *
from .block import __all__ as _block_all

__all__ += _block_all