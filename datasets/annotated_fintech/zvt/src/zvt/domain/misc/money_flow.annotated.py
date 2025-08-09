# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, Float

# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin

# 🧠 ML Signal: Inheritance from MoneyFlowBase and Mixin indicates a pattern of using base classes and mixins for shared functionality
from zvt.contract.register import register_schema

# ✅ Best Practice: Naming convention for base classes should be clear and descriptive.

# 🧠 ML Signal: Use of __tablename__ suggests a pattern of ORM usage for database table mapping
MoneyFlowBase = declarative_base()

# 🧠 ML Signal: Use of Column with String type indicates a pattern of defining database schema with specific data types

#: 板块资金流向
# 🧠 ML Signal: Use of Column with String type indicates a pattern of defining database schema with specific data types


# 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
class BlockMoneyFlow(MoneyFlowBase, Mixin):
    __tablename__ = "block_money_flow"
    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types

    code = Column(String(length=32))
    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    name = Column(String(length=32))

    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    #: 收盘价
    close = Column(Float)
    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    change_pct = Column(Float)
    turnover_rate = Column(Float)
    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types

    #: 净流入
    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    net_inflows = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 净流入率
    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    net_inflow_rate = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction

    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    #: 主力=超大单+大单
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    net_main_inflows = Column(Float)
    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    net_main_inflow_rate = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 超大单
    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    net_huge_inflows = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    net_huge_inflow_rate = Column(Float)
    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    #: 大单
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    net_big_inflows = Column(Float)
    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    net_big_inflow_rate = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction

    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    #: 中单
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    net_medium_inflows = Column(Float)
    # 🧠 ML Signal: Use of Column with Float type indicates a pattern of defining database schema with specific data types
    net_medium_inflow_rate = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 小单
    net_small_inflows = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    net_small_inflow_rate = Column(Float)


# ✅ Best Practice: Define column types and constraints for database schema

# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction


# ✅ Best Practice: Define column types and constraints for database schema
class StockMoneyFlow(MoneyFlowBase, Mixin):
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    __tablename__ = "stock_money_flow"
    # ✅ Best Practice: Define column types and constraints for database schema

    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    code = Column(String(length=32))
    # ✅ Best Practice: Define column types and constraints for database schema
    name = Column(String(length=32))
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction

    # ✅ Best Practice: Define column types and constraints for database schema
    #: 收盘价
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    close = Column(Float)
    # ✅ Best Practice: Define column types and constraints for database schema
    change_pct = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    turnover_rate = Column(Float)
    # ✅ Best Practice: Define column types and constraints for database schema

    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 净流入
    # ✅ Best Practice: Define column types and constraints for database schema
    net_inflows = Column(Float)
    # 🧠 ML Signal: Use of __all__ to define public API of the module
    # ✅ Best Practice: Define column types and constraints for database schema
    # 🧠 ML Signal: Usage of register_schema function indicates schema registration pattern
    #: 净流入率
    net_inflow_rate = Column(Float)

    #: 主力=超大单+大单
    net_main_inflows = Column(Float)
    net_main_inflow_rate = Column(Float)
    #: 超大单
    net_huge_inflows = Column(Float)
    net_huge_inflow_rate = Column(Float)
    #: 大单
    net_big_inflows = Column(Float)
    net_big_inflow_rate = Column(Float)

    #: 中单
    net_medium_inflows = Column(Float)
    net_medium_inflow_rate = Column(Float)
    #: 小单
    net_small_inflows = Column(Float)
    net_small_inflow_rate = Column(Float)


class IndexMoneyFlow(MoneyFlowBase, Mixin):
    __tablename__ = "index_money_flow"

    code = Column(String(length=32))
    name = Column(String(length=32))

    #: 净流入
    net_inflows = Column(Float)
    #: 净流入率
    net_inflow_rate = Column(Float)

    #: 主力=超大单+大单
    net_main_inflows = Column(Float)
    net_main_inflow_rate = Column(Float)
    #: 超大单
    net_huge_inflows = Column(Float)
    net_huge_inflow_rate = Column(Float)
    #: 大单
    net_big_inflows = Column(Float)
    net_big_inflow_rate = Column(Float)

    #: 中单
    net_medium_inflows = Column(Float)
    net_medium_inflow_rate = Column(Float)
    #: 小单
    net_small_inflows = Column(Float)
    net_small_inflow_rate = Column(Float)


register_schema(
    providers=["joinquant", "sina"],
    db_name="money_flow",
    schema_base=MoneyFlowBase,
    entity_type="stock",
)


# the __all__ is generated
__all__ = ["BlockMoneyFlow", "StockMoneyFlow", "IndexMoneyFlow"]
