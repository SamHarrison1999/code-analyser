# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, DateTime, Float

# ✅ Best Practice: Group related imports together for better readability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin
from zvt.contract.register import register_schema

# ✅ Best Practice: Naming convention for base classes should be clear and descriptive.
# ✅ Best Practice: Define a class-level variable for the table name to avoid magic strings

HolderBase = declarative_base()
# ✅ Best Practice: Specify a maximum length for string columns to prevent excessive data storage


# ✅ Best Practice: Specify a maximum length for string columns to prevent excessive data storage
class HkHolder(HolderBase, Mixin):
    __tablename__ = "hk_holder"
    # ✅ Best Practice: Specify a maximum length for string columns to prevent excessive data storage
    #: 股票代码
    code = Column(String(length=32))
    # ✅ Best Practice: Specify a maximum length for string columns to prevent excessive data storage
    # 🧠 ML Signal: Inheritance from multiple classes indicates a pattern of using mixins or base classes.
    #: 股票名称
    name = Column(String(length=32))
    # ⚠️ SAST Risk (Low): Ensure that the Float type is appropriate for financial data to avoid precision issues
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database table mapping.

    #: 市场通编码	三种类型：310001-沪股通，310002-深股通，310005-港股通
    # ⚠️ SAST Risk (Low): Ensure that the Float type is appropriate for financial data to avoid precision issues
    # ✅ Best Practice: Specifying length for String columns improves database performance and storage.
    holder_code = Column(String(length=32))
    #: 市场通名称	三种类型：沪股通，深股通，港股通
    # ✅ Best Practice: Specifying length for String columns improves database performance and storage.
    holder_name = Column(String(length=32))

    # ✅ Best Practice: Specifying length for String columns improves database performance and storage.
    #: 持股数量
    share_number = Column(Float)
    # ✅ Best Practice: Use of DateTime for date fields ensures proper date handling and querying.
    #: 持股比例
    # 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchy and relationships
    share_ratio = Column(Float)


# ✅ Best Practice: Specifying length for String columns improves database performance and storage.

# 🧠 ML Signal: Use of class variable for table name, indicating ORM pattern


# ✅ Best Practice: Specifying length for String columns improves database performance and storage.
class TopTenTradableHolder(HolderBase, Mixin):
    # 🧠 ML Signal: Definition of database columns, useful for schema inference
    __tablename__ = "top_ten_tradable_holder"
    # ✅ Best Practice: Use of Float for numeric fields allows for decimal precision.

    # 🧠 ML Signal: Definition of database columns, useful for schema inference
    provider = Column(String(length=32))
    # ✅ Best Practice: Use of Float for numeric fields allows for decimal precision.
    code = Column(String(length=32))
    # 🧠 ML Signal: Definition of database columns, useful for schema inference

    # ✅ Best Practice: Use of Float for numeric fields allows for decimal precision.
    report_period = Column(String(length=32))
    # 🧠 ML Signal: Definition of database columns, useful for schema inference
    report_date = Column(DateTime)
    # ✅ Best Practice: Use of Float for numeric fields allows for decimal precision.

    # 🧠 ML Signal: Definition of database columns, useful for schema inference
    #: 股东代码
    holder_code = Column(String(length=32))
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    # 🧠 ML Signal: Definition of database columns, useful for schema inference
    #: 股东名称
    holder_name = Column(String(length=32))
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    # 🧠 ML Signal: Definition of database columns, useful for schema inference
    #: 持股数
    shareholding_numbers = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    # 🧠 ML Signal: Definition of database columns, useful for schema inference
    #: 持股比例
    shareholding_ratio = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    # 🧠 ML Signal: Definition of database columns, useful for schema inference
    #: 变动
    change = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    # 🧠 ML Signal: Definition of database columns, useful for schema inference
    #: 变动比例
    change_ratio = Column(Float)


# 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
# 🧠 ML Signal: Registration of schema with specific providers and database
# ✅ Best Practice: Explicitly specifying providers and database details for schema registration
# ✅ Best Practice: Use of __all__ to define public API of the module


class TopTenHolder(HolderBase, Mixin):
    __tablename__ = "top_ten_holder"

    provider = Column(String(length=32))
    code = Column(String(length=32))

    report_period = Column(String(length=32))
    report_date = Column(DateTime)

    #: 股东代码
    holder_code = Column(String(length=32))
    #: 股东名称
    holder_name = Column(String(length=32))
    #: 持股数
    shareholding_numbers = Column(Float)
    #: 持股比例
    shareholding_ratio = Column(Float)
    #: 变动
    change = Column(Float)
    #: 变动比例
    change_ratio = Column(Float)


class InstitutionalInvestorHolder(HolderBase, Mixin):
    __tablename__ = "institutional_investor_holder"

    provider = Column(String(length=32))
    code = Column(String(length=32))

    report_period = Column(String(length=32))
    report_date = Column(DateTime)

    #: 机构类型
    institutional_investor_type = Column(String(length=64))
    #: 股东代码
    holder_code = Column(String(length=32))
    #: 股东名称
    holder_name = Column(String(length=32))
    #: 持股数
    shareholding_numbers = Column(Float)
    #: 持股比例
    shareholding_ratio = Column(Float)


register_schema(
    providers=["eastmoney", "joinquant"],
    db_name="holder",
    schema_base=HolderBase,
    entity_type="stock",
)


# the __all__ is generated
__all__ = [
    "HkHolder",
    "TopTenTradableHolder",
    "TopTenHolder",
    "InstitutionalInvestorHolder",
]
