# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, DateTime, Float, Boolean, Integer
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract.register import register_schema
# ✅ Best Practice: Using declarative_base() to create a base class for declarative class definitions is a standard practice in SQLAlchemy.
from zvt.contract.schema import TradableMeetActor
# ✅ Best Practice: Use of class variable for table name ensures consistency and easy maintenance

StockActorBase = declarative_base()
# ✅ Best Practice: Use of specific column types improves readability and enforces data integrity


# ✅ Best Practice: Use of DateTime for report_date ensures proper handling of date and time data
class StockTopTenFreeHolder(StockActorBase, TradableMeetActor):
    __tablename__ = "stock_top_ten_free_holder"
    # ✅ Best Practice: Use of Float for holding_numbers allows for precise representation of numerical data

    # ✅ Best Practice: Define a table name for ORM mapping
    report_period = Column(String(length=32))
    # ✅ Best Practice: Use of Float for holding_ratio allows for precise representation of numerical data
    report_date = Column(DateTime)
    # ✅ Best Practice: Use descriptive column names for clarity

    # ✅ Best Practice: Use of Float for holding_values allows for precise representation of numerical data
    #: 持股数
    # ✅ Best Practice: Use DateTime for date-related fields for consistency
    holding_numbers = Column(Float)
    #: 持股比例
    # ✅ Best Practice: Use Float for numerical fields that may have decimal values
    holding_ratio = Column(Float)
    # 🧠 ML Signal: Inheritance from multiple base classes indicates a design pattern that could be learned.
    #: 持股市值
    # ✅ Best Practice: Use Float for numerical fields that may have decimal values
    holding_values = Column(Float)
# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction is a common pattern.

# ✅ Best Practice: Use Float for numerical fields that may have decimal values

# ✅ Best Practice: Specifying length for String type improves database schema design.
class StockTopTenHolder(StockActorBase, TradableMeetActor):
    __tablename__ = "stock_top_ten_holder"
    # ⚠️ SAST Risk (Low): Ensure that DateTime is timezone-aware to prevent timezone-related issues.

    report_period = Column(String(length=32))
    # ⚠️ SAST Risk (Low): Floating point numbers can introduce precision issues; consider using Decimal for financial data.
    # ✅ Best Practice: Define column types with explicit lengths for better database schema management
    report_date = Column(DateTime)

    # ⚠️ SAST Risk (Low): Floating point numbers can introduce precision issues; consider using Decimal for financial data.
    # ✅ Best Practice: Define column types with explicit lengths for better database schema management
    #: 持股数
    holding_numbers = Column(Float)
    # ⚠️ SAST Risk (Low): Floating point numbers can introduce precision issues; consider using Decimal for financial data.
    # ✅ Best Practice: Define column types with explicit lengths for better database schema management
    #: 持股比例
    holding_ratio = Column(Float)
    #: 持股市值
    holding_values = Column(Float)


class StockInstitutionalInvestorHolder(StockActorBase, TradableMeetActor):
    __tablename__ = "stock_institutional_investor_holder"
    # 🧠 ML Signal: Use of __all__ to define public API of the module
    # 🧠 ML Signal: Usage of register_schema function indicates a pattern for schema registration

    report_period = Column(String(length=32))
    report_date = Column(DateTime)

    #: 持股数
    holding_numbers = Column(Float)
    #: 持股比例
    holding_ratio = Column(Float)
    #: 持股市值
    holding_values = Column(Float)


class StockActorSummary(StockActorBase, TradableMeetActor):
    __tablename__ = "stock_actor_summary"
    #: tradable code
    code = Column(String(length=64))
    #: tradable name
    name = Column(String(length=128))

    report_period = Column(String(length=32))
    report_date = Column(DateTime)

    #: 变动比例
    change_ratio = Column(Float)
    #: 是否完成
    is_complete = Column(Boolean)
    #: 持股市值
    actor_type = Column(String)
    actor_count = Column(Integer)

    #: 持股数
    holding_numbers = Column(Float)
    #: 持股比例
    holding_ratio = Column(Float)
    #: 持股市值
    holding_values = Column(Float)


register_schema(providers=["em"], db_name="stock_actor", schema_base=StockActorBase, entity_type="stock")


# the __all__ is generated
__all__ = ["StockTopTenFreeHolder", "StockTopTenHolder", "StockInstitutionalInvestorHolder", "StockActorSummary"]