# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, Float
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin
from zvt.contract.register import register_schema
# ✅ Best Practice: Naming convention for classes should follow CamelCase.
# ✅ Best Practice: Define a class-level attribute for the table name to ensure consistency and easy maintenance.

TradingBase = declarative_base()
# 🧠 ML Signal: Usage of SQLAlchemy's Column and String types indicates ORM pattern for database interaction.


# 🧠 ML Signal: Consistent use of String type with specified length for text fields.
class ManagerTrading(TradingBase, Mixin):
    __tablename__ = "manager_trading"
    # 🧠 ML Signal: Consistent naming pattern for attributes related to trading.

    provider = Column(String(length=32))
    # 🧠 ML Signal: Use of Float type for numerical fields indicates handling of decimal values.
    code = Column(String(length=32))
    #: 日期 变动人 变动数量(股) 交易均价(元) 结存股票(股) 交易方式 董监高管 高管职位 与高管关系
    # 🧠 ML Signal: Consistent use of Float type for financial data.
    #: 2017-08-11 韦春 200 9.16 -- 竞价交易 刘韬 高管 兄弟姐妹

    # 🧠 ML Signal: Attribute naming suggests this field tracks the amount of holdings.
    # 🧠 ML Signal: Inheritance from multiple classes indicates a pattern of using mixins or base classes for shared functionality.
    #: 变动人
    trading_person = Column(String(length=32))
    # 🧠 ML Signal: Use of String type for categorical data.
    # 🧠 ML Signal: Use of SQLAlchemy ORM pattern for database table representation.
    #: 变动数量
    volume = Column(Float)
    # 🧠 ML Signal: Attribute naming suggests this field tracks manager information.
    # 🧠 ML Signal: Use of fixed-length strings for database columns, indicating a pattern of data size constraints.
    #: 交易均价
    price = Column(Float)
    # 🧠 ML Signal: Consistent naming pattern for attributes related to manager details.
    # 🧠 ML Signal: Use of fixed-length strings for database columns, indicating a pattern of data size constraints.
    #: 结存股票
    holding = Column(Float)
    # 🧠 ML Signal: Attribute naming suggests this field tracks relationships, which could be sensitive information.
    # 🧠 ML Signal: Use of fixed-length strings for database columns, indicating a pattern of data size constraints.
    # ✅ Best Practice: Use of class variable for table name ensures consistency and easy maintenance
    #: 交易方式
    trading_way = Column(String(length=32))
    # 🧠 ML Signal: Use of Float type for numerical data, indicating a pattern of handling decimal values.
    # ✅ Best Practice: Specifying column types and lengths improves database schema clarity
    #: 董监高管
    manager = Column(String(length=32))
    # 🧠 ML Signal: Use of Float type for numerical data, indicating a pattern of handling decimal values.
    # ✅ Best Practice: Specifying column types and lengths improves database schema clarity
    #: 高管职位
    manager_position = Column(String(length=32))
    # 🧠 ML Signal: Use of Float type for numerical data, indicating a pattern of handling decimal values.
    # ✅ Best Practice: Specifying column types improves database schema clarity
    #: 与高管关系
    relationship_with_manager = Column(String(length=32))
# ✅ Best Practice: Specifying column types improves database schema clarity
# ✅ Best Practice: Class name should be descriptive and follow CamelCase naming convention


# ✅ Best Practice: Specifying column types and lengths improves database schema clarity
# ✅ Best Practice: Use a class variable to define the table name for ORM mapping
class HolderTrading(TradingBase, Mixin):
    __tablename__ = "holder_trading"
    # ✅ Best Practice: Specifying column types and lengths improves database schema clarity
    # ✅ Best Practice: Use descriptive column names and specify data types for ORM mapping

    provider = Column(String(length=32))
    # ✅ Best Practice: Specifying column types improves database schema clarity
    # ✅ Best Practice: Use descriptive column names and specify data types for ORM mapping
    code = Column(String(length=32))

    # ✅ Best Practice: Use descriptive column names and specify data types for ORM mapping
    #: 股东名称
    holder_name = Column(String(length=32))
    # ✅ Best Practice: Use descriptive column names and specify data types for ORM mapping
    #: 变动数量
    volume = Column(Float)
    # ✅ Best Practice: Use descriptive column names and specify data types for ORM mapping
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 变动比例
    change_pct = Column(Float)
    # ✅ Best Practice: Use descriptive column names and specify data types for ORM mapping
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 变动后持股比例
    holding_pct = Column(Float)
# ✅ Best Practice: Use descriptive column names and specify data types for ORM mapping
# 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling


# ✅ Best Practice: Use descriptive column names and specify data types for ORM mapping
# 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
class BigDealTrading(TradingBase, Mixin):
    __tablename__ = "big_deal_trading"
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling

    provider = Column(String(length=32))
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    code = Column(String(length=32))

    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 成交额
    turnover = Column(Float)
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 成交价
    price = Column(Float)
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 卖出营业部
    sell_broker = Column(String(length=128))
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 买入营业部
    buy_broker = Column(String(length=128))
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 折/溢价率
    compare_rate = Column(Float)
# 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling


# 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
class MarginTrading(TradingBase, Mixin):
    __tablename__ = "margin_trading"
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    code = Column(String(length=32))

    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 融资余额(元）
    fin_value = Column(Float)
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 融资买入额（元）
    fin_buy_value = Column(Float)
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 融资偿还额（元）
    fin_refund_value = Column(Float)
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 融券余量（股）
    sec_value = Column(Float)
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 融券卖出量（股）
    sec_sell_value = Column(Float)
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 融券偿还量（股）
    sec_refund_value = Column(Float)
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    #: 融资融券余额（元）
    fin_sec_value = Column(Float)
# 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling


# 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
class DragonAndTiger(TradingBase, Mixin):
    __tablename__ = "dragon_and_tiger"
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling

    code = Column(String(length=32))
    name = Column(String(length=32))
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    # 🧠 ML Signal: Registration of schema with specific providers and database details
    # 🧠 ML Signal: Definition of module exports

    #: 异动原因
    reason = Column(String(length=128))
    #: 成交额
    turnover = Column(Float)
    #: 涨幅
    change_pct = Column(Float)
    #: 净买入
    net_in = Column(Float)

    #: 买入营业部
    dep1 = Column(String(length=128))
    dep1_in = Column(Float)
    dep1_out = Column(Float)
    dep1_rate = Column(Float)

    dep2 = Column(String(length=128))
    dep2_in = Column(Float)
    dep2_out = Column(Float)
    dep2_rate = Column(Float)

    dep3 = Column(String(length=128))
    dep3_in = Column(Float)
    dep3_out = Column(Float)
    dep3_rate = Column(Float)

    dep4 = Column(String(length=128))
    dep4_in = Column(Float)
    dep4_out = Column(Float)
    dep4_rate = Column(Float)

    dep5 = Column(String(length=128))
    dep5_in = Column(Float)
    dep5_out = Column(Float)
    dep5_rate = Column(Float)

    #: 卖出营业部
    dep_1 = Column(String(length=128))
    dep_1_in = Column(Float)
    dep_1_out = Column(Float)
    dep_1_rate = Column(Float)

    dep_2 = Column(String(length=128))
    dep_2_in = Column(Float)
    dep_2_out = Column(Float)
    dep_2_rate = Column(Float)

    dep_3 = Column(String(length=128))
    dep_3_in = Column(Float)
    dep_3_out = Column(Float)
    dep_3_rate = Column(Float)

    dep_4 = Column(String(length=128))
    dep_4_in = Column(Float)
    dep_4_out = Column(Float)
    dep_4_rate = Column(Float)

    dep_5 = Column(String(length=128))
    dep_5_in = Column(Float)
    dep_5_out = Column(Float)
    dep_5_rate = Column(Float)


register_schema(
    providers=["em", "eastmoney", "joinquant"], db_name="trading", schema_base=TradingBase, entity_type="stock"
)


# the __all__ is generated
__all__ = ["ManagerTrading", "HolderTrading", "BigDealTrading", "MarginTrading", "DragonAndTiger"]