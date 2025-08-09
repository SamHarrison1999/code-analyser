# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, Float
# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin
# ✅ Best Practice: Naming the base class as OverallBase provides clarity on its purpose.
from zvt.contract.register import register_schema
# 🧠 ML Signal: Inheritance from OverallBase and Mixin suggests a pattern for class design

OverallBase = declarative_base()
# 🧠 ML Signal: Use of SQLAlchemy ORM for database table mapping


# ✅ Best Practice: Specify length for String columns to optimize database storage
#: 市场整体估值

# ✅ Best Practice: Specify length for String columns to optimize database storage

class StockSummary(OverallBase, Mixin):
    # ✅ Best Practice: Specify length for String columns to optimize database storage
    __tablename__ = "stock_summary"

    # ✅ Best Practice: Use of Float for numerical data to handle large and small numbers
    provider = Column(String(length=32))
    code = Column(String(length=32))
    # ⚠️ SAST Risk (Low): Potential typo in 'total_tradable_vaule', should be 'total_tradable_value'
    name = Column(String(length=32))
    # ✅ Best Practice: Define column types and constraints for database schema

    # ✅ Best Practice: Use of Float for numerical data to handle large and small numbers
    total_value = Column(Float)
    # ✅ Best Practice: Define column types and constraints for database schema
    total_tradable_vaule = Column(Float)
    # ✅ Best Practice: Use of Float for numerical data to handle large and small numbers
    pe = Column(Float)
    # ✅ Best Practice: Define column types and constraints for database schema
    pb = Column(Float)
    # ✅ Best Practice: Use of Float for numerical data to handle large and small numbers
    volume = Column(Float)
    # ✅ Best Practice: Define column types and constraints for database schema
    turnover = Column(Float)
    # ✅ Best Practice: Use of Float for numerical data to handle large and small numbers
    turnover_rate = Column(Float)
# ✅ Best Practice: Define column types and constraints for database schema

# ✅ Best Practice: Use of Float for numerical data to handle large and small numbers

# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
# ✅ Best Practice: Define column types and constraints for database schema
#: 融资融券概况

# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
# ✅ Best Practice: Define column types and constraints for database schema

class MarginTradingSummary(OverallBase, Mixin):
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    # ✅ Best Practice: Define column types and constraints for database schema
    __tablename__ = "margin_trading_summary"
    provider = Column(String(length=32))
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    code = Column(String(length=32))
    name = Column(String(length=32))
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction

    #: 融资余额
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    # 🧠 ML Signal: Registration of schema with specific providers and database
    # ✅ Best Practice: Use of __all__ to define public API of the module
    margin_value = Column(Float)
    #: 买入额
    margin_buy = Column(Float)

    #: 融券余额
    short_value = Column(Float)
    #: 卖出量
    short_volume = Column(Float)

    #: 融资融券余额
    total_value = Column(Float)


#: 北向/南向成交概况


class CrossMarketSummary(OverallBase, Mixin):
    __tablename__ = "cross_market_summary"
    provider = Column(String(length=32))
    code = Column(String(length=32))
    name = Column(String(length=32))

    buy_amount = Column(Float)
    buy_volume = Column(Float)
    sell_amount = Column(Float)
    sell_volume = Column(Float)
    quota_daily = Column(Float)
    quota_daily_balance = Column(Float)


register_schema(providers=["joinquant", "exchange"], db_name="overall", schema_base=OverallBase, entity_type="stock")


# the __all__ is generated
__all__ = ["StockSummary", "MarginTradingSummary", "CrossMarketSummary"]