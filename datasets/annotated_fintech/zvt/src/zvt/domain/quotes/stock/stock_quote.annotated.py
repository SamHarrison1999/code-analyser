# -*- coding: utf-8 -*-
from sqlalchemy import String, Column, Float, Integer, Boolean, JSON

# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin
from zvt.contract.register import register_schema

# ✅ Best Practice: Naming convention for base classes should be consistent and descriptive.
from zvt.domain.quotes import StockKdataCommon

# 🧠 ML Signal: Use of a string column for 'code' indicates categorical data that could be used for classification tasks.
StockQuoteBase = declarative_base()

# 🧠 ML Signal: 'time' as an integer suggests a timestamp, which is often used in time series analysis.


class StockTick(StockQuoteBase, Mixin):
    # 🧠 ML Signal: 'lastPrice' as a float is a continuous variable, useful for regression models.
    __tablename__ = "stock_tick"

    # 🧠 ML Signal: 'open', 'high', 'low', 'lastClose', 'amount', 'volume', 'pvolume' are continuous variables, useful for regression models.
    code = Column(String(length=32))

    #: UNIX时间戳
    time = Column(Integer)
    #: 最新价
    lastPrice = Column(Float)

    # 开盘价
    # 🧠 ML Signal: Use of JSON columns for 'askPrice', 'askVol', 'bidPrice', 'bidVol' indicates complex data structures, useful for feature extraction.
    open = Column(Float)
    # 最高价
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    high = Column(Float)
    # 最低价
    # 🧠 ML Signal: Storing price as a float, indicating financial data
    low = Column(Float)
    # 上日收盘价
    # 🧠 ML Signal: Boolean flags for stock limit status
    lastClose = Column(Float)

    # 🧠 ML Signal: Financial threshold values for stock limits
    amount = Column(Float)
    volume = Column(Float)
    # 🧠 ML Signal: Boolean flags for stock limit status
    pvolume = Column(Float)

    # 🧠 ML Signal: Financial threshold values for stock limits
    askPrice = Column(JSON)
    # ✅ Best Practice: Define __tablename__ to explicitly specify the table name in the database.
    askVol = Column(JSON)
    # 🧠 ML Signal: Financial data related to stock trading
    bidPrice = Column(JSON)
    # 🧠 ML Signal: Integer type for time suggests it might be a timestamp.
    bidVol = Column(JSON)


# 🧠 ML Signal: Financial data related to stock trading

# 🧠 ML Signal: Float type for price indicates it is a continuous value.


# 🧠 ML Signal: Financial metrics for stock valuation
class StockQuote(StockQuoteBase, StockKdataCommon):
    # 🧠 ML Signal: Boolean flags for limit up/down can be used for classification tasks.
    __tablename__ = "stock_quote"
    # 🧠 ML Signal: Financial metrics for stock valuation
    #: UNIX时间戳
    # 🧠 ML Signal: Float type for limit up amount indicates it is a continuous value.
    time = Column(Integer)
    #: 最新价
    # 🧠 ML Signal: Boolean flags for limit up/down can be used for classification tasks.
    price = Column(Float)
    #: 是否涨停
    # 🧠 ML Signal: Float type for limit down amount indicates it is a continuous value.
    is_limit_up = Column(Boolean)
    # 🧠 ML Signal: Use of a fixed-length string for 'code' suggests a pattern for stock identifiers
    #: 封涨停金额
    # 🧠 ML Signal: Float type for ask amount indicates it is a continuous value.
    limit_up_amount = Column(Float)
    # 🧠 ML Signal: Use of a fixed-length string for 'name' suggests a pattern for stock names
    #: 是否跌停
    # 🧠 ML Signal: Float type for bid amount indicates it is a continuous value.
    is_limit_down = Column(Boolean)
    # 🧠 ML Signal: Use of Integer for 'time' suggests a pattern for timestamp or time representation
    #: 封跌停金额
    # 🧠 ML Signal: Float type for float cap indicates it is a continuous value.
    limit_down_amount = Column(Float)
    # 🧠 ML Signal: Use of Float for 'price' suggests a pattern for financial data representation
    #: 5挡卖单金额
    # 🧠 ML Signal: Float type for total cap indicates it is a continuous value.
    ask_amount = Column(Float)
    # 🧠 ML Signal: Use of Float for 'avg_price' suggests a pattern for financial data representation
    #: 5挡买单金额
    bid_amount = Column(Float)
    # 🧠 ML Signal: Use of Float for 'change_pct' suggests a pattern for percentage change representation
    #: 流通市值
    float_cap = Column(Float)
    # ✅ Best Practice: Use of '__all__' to define public API of the module
    # 🧠 ML Signal: Use of Float for 'volume' suggests a pattern for financial data representation
    # 🧠 ML Signal: Use of Float for 'turnover_rate' suggests a pattern for percentage or rate representation
    # 🧠 ML Signal: Use of Boolean for 'is_limit_up' suggests a pattern for binary state representation
    # ⚠️ SAST Risk (Low): Potential risk if 'register_schema' is not properly validated or sanitized
    #: 总市值
    total_cap = Column(Float)


class StockQuoteLog(StockQuoteBase, StockKdataCommon):
    __tablename__ = "stock_quote_log"
    #: UNIX时间戳
    time = Column(Integer)
    #: 最新价
    price = Column(Float)
    #: 是否涨停
    is_limit_up = Column(Boolean)
    #: 封涨停金额
    limit_up_amount = Column(Float)
    #: 是否跌停
    is_limit_down = Column(Boolean)
    #: 封跌停金额
    limit_down_amount = Column(Float)
    #: 5挡卖单金额
    ask_amount = Column(Float)
    #: 5挡买单金额
    bid_amount = Column(Float)
    #: 流通市值
    float_cap = Column(Float)
    #: 总市值
    total_cap = Column(Float)


class Stock1mQuote(StockQuoteBase, Mixin):
    __tablename__ = "stock_1m_quote"
    code = Column(String(length=32))
    name = Column(String(length=32))

    #: UNIX时间戳
    time = Column(Integer)
    #: 最新价
    price = Column(Float)
    #: 均价
    avg_price = Column(Float)
    # 涨跌幅
    change_pct = Column(Float)
    # 成交量
    volume = Column(Float)
    # 成交金额
    turnover = Column(Float)
    # 换手率
    turnover_rate = Column(Float)
    #: 是否涨停
    is_limit_up = Column(Boolean)
    #: 是否跌停
    is_limit_down = Column(Boolean)


register_schema(
    providers=["qmt"],
    db_name="stock_quote",
    schema_base=StockQuoteBase,
    entity_type="stock",
)


# the __all__ is generated
__all__ = ["StockQuote", "StockQuoteLog", "Stock1mQuote"]
