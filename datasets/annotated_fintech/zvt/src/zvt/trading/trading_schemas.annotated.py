# -*- coding: utf-8 -*-
from sqlalchemy import Column, Float, DateTime, Integer
from sqlalchemy import String, JSON

# ✅ Best Practice: Grouping related imports together improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin

# ✅ Best Practice: Naming convention for base classes should be consistent and descriptive.
from zvt.contract.register import register_schema

# ✅ Best Practice: Use of class-level variable for table name improves maintainability and readability.

TradingBase = declarative_base()
# ✅ Best Practice: Defining columns with types improves readability and ensures correct data handling.


# ✅ Best Practice: Defining columns with types improves readability and ensures correct data handling.
class TagQuoteStats(Mixin, TradingBase):
    __tablename__ = "tag_quote_stats"
    # ✅ Best Practice: Defining columns with types improves readability and ensures correct data handling.
    stock_pool_name = Column(String)
    main_tag = Column(String)
    # ✅ Best Practice: Defining columns with types improves readability and ensures correct data handling.
    limit_up_count = Column(Integer)
    # 🧠 ML Signal: Inheritance from TradingBase and Mixin indicates a pattern for extending functionality
    limit_down_count = Column(Integer)
    # ✅ Best Practice: Defining columns with types improves readability and ensures correct data handling.
    up_count = Column(Integer)
    # 🧠 ML Signal: Use of __tablename__ suggests ORM pattern for database interaction
    down_count = Column(Integer)
    # ✅ Best Practice: Defining columns with types improves readability and ensures correct data handling.
    change_pct = Column(Float)
    # 🧠 ML Signal: Column definitions indicate a pattern for database schema design
    turnover = Column(Float)


# ✅ Best Practice: Defining columns with types improves readability and ensures correct data handling.

# 🧠 ML Signal: Column definitions indicate a pattern for database schema design


# ✅ Best Practice: Defining columns with types improves readability and ensures correct data handling.
class TradingPlan(TradingBase, Mixin):
    # 🧠 ML Signal: Column definitions indicate a pattern for database schema design
    __tablename__ = "trading_plan"
    stock_id = Column(String)
    # 🧠 ML Signal: Column definitions indicate a pattern for database schema design
    stock_code = Column(String)
    stock_name = Column(String)
    # 🧠 ML Signal: Column definitions indicate a pattern for database schema design
    trading_date = Column(DateTime)
    # ⚠️ SAST Risk (Low): Nullable=False on expected_open_pct could lead to integrity errors if not handled
    # 预期开盘涨跌幅
    # 🧠 ML Signal: Inheritance from multiple classes indicates a pattern of using mixins for shared functionality.
    expected_open_pct = Column(Float, nullable=False)
    # 🧠 ML Signal: Column definitions indicate a pattern for database schema design
    buy_price = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM to define database table structure.
    sell_price = Column(Float)
    # 🧠 ML Signal: Column definitions indicate a pattern for database schema design
    # 操作理由
    # 🧠 ML Signal: Column definitions indicate a pattern for database schema design
    # 🧠 ML Signal: Use of __all__ to define public API of the module.
    # 🧠 ML Signal: Use of JSON type in SQLAlchemy for storing complex data structures.
    # ⚠️ SAST Risk (Low): Ensure that the database schema is properly validated to prevent SQL injection.
    trading_reason = Column(String)
    # 交易信号
    trading_signal_type = Column(String)
    # 执行状态
    status = Column(String)
    # 复盘
    review = Column(String)


class QueryStockQuoteSetting(TradingBase, Mixin):
    __tablename__ = "query_stock_quote_setting"
    stock_pool_name = Column(String)
    main_tags = Column(JSON)


register_schema(providers=["zvt"], db_name="stock_trading", schema_base=TradingBase)


# the __all__ is generated
__all__ = ["TradingPlan", "QueryStockQuoteSetting"]
