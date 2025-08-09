# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, DateTime, Boolean, Float, Integer, ForeignKey
from sqlalchemy.orm import declarative_base

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy.orm import relationship

from zvt.contract import Mixin
from zvt.contract.register import register_schema

# ✅ Best Practice: Naming convention for base classes should be consistent and descriptive.
from zvt.utils.decorator import to_string

TraderBase = declarative_base()

# ✅ Best Practice: Use of a class variable to define the table name for ORM mapping


class TraderInfo(TraderBase, Mixin):
    """
    trader info
    # ✅ Best Practice: Use of Column with String type and length for database schema definition
    """

    # ✅ Best Practice: Use of Column with DateTime type for database schema definition
    __tablename__ = "trader_info"
    #: 机器人名字
    # ✅ Best Practice: Use of Column with DateTime type for database schema definition
    trader_name = Column(String(length=128))
    # ✅ Best Practice: Use of Column with String type and length for database schema definition

    entity_type = Column(String(length=128))
    start_timestamp = Column(DateTime)
    # ✅ Best Practice: Use of Column with String type and length for database schema definition
    end_timestamp = Column(DateTime)
    provider = Column(String(length=32))
    level = Column(String(length=32))
    # ✅ Best Practice: Define a clear and descriptive table name for ORM mapping
    # ✅ Best Practice: Use of Column with Boolean type for database schema definition
    real_time = Column(Boolean)
    # ✅ Best Practice: Use of Column with Boolean type for database schema definition
    kdata_use_begin_time = Column(Boolean)
    # 🧠 ML Signal: Usage of financial attributes for account statistics
    kdata_adjust_type = Column(String(length=32))


# ✅ Best Practice: Use of Column with String type and length for database schema definition

# 🧠 ML Signal: Usage of trader name as a string attribute


# 🧠 ML Signal: Use of decorator pattern to enhance or modify class behavior
@to_string
# 🧠 ML Signal: Usage of cash as a financial attribute
class AccountStats(TraderBase, Mixin):
    """
    account stats of every day
    """

    # 🧠 ML Signal: Usage of value as a financial attribute
    # 🧠 ML Signal: Inheritance from TraderBase and Mixin indicates a pattern of using mixins for shared functionality

    __tablename__ = "account_stats"
    # 🧠 ML Signal: Usage of all_value as a financial attribute
    # 🧠 ML Signal: Use of __tablename__ suggests ORM pattern for database table mapping

    input_money = Column(Float)
    # 🧠 ML Signal: Usage of profit as a financial attribute
    # 🧠 ML Signal: Use of Column and String indicates ORM pattern for defining database schema

    #: 机器人名字
    # 🧠 ML Signal: Usage of profit_rate as a financial attribute
    # 🧠 ML Signal: Use of ForeignKey indicates a relational database pattern
    trader_name = Column(String(length=128))
    #: 可用现金
    # 🧠 ML Signal: Usage of closing as a boolean attribute to indicate state
    # 🧠 ML Signal: Use of relationship indicates ORM pattern for defining relationships between tables
    cash = Column(Float)
    #: 具体仓位
    # 🧠 ML Signal: Use of Float for financial data suggests a pattern of handling monetary values
    positions = relationship("Position", back_populates="account_stats")
    #: 市值
    # 🧠 ML Signal: Use of Float for financial data suggests a pattern of handling monetary values
    value = Column(Float)
    #: 市值+cash
    # 🧠 ML Signal: Use of Float for financial data suggests a pattern of handling monetary values
    all_value = Column(Float)

    # ✅ Best Practice: Use of __tablename__ for ORM class to specify the database table name
    # 🧠 ML Signal: Use of Float for financial data suggests a pattern of handling monetary values
    #: 盈亏
    profit = Column(Float)
    # 🧠 ML Signal: Use of Float for financial data suggests a pattern of handling monetary values
    # ✅ Best Practice: Use of Column with String type for trader_name ensures proper database schema definition
    #: 盈亏比例
    profit_rate = Column(Float)
    # 🧠 ML Signal: Use of Float for financial data suggests a pattern of handling monetary values
    # ✅ Best Practice: Use of Column with Float type for order_price ensures proper database schema definition

    #: 收盘计算
    # 🧠 ML Signal: Use of Float for financial data suggests a pattern of handling monetary values
    # ✅ Best Practice: Use of Column with Float type for order_amount ensures proper database schema definition
    closing = Column(Boolean)


# ✅ Best Practice: Use of __all__ to define public API of the module
# 🧠 ML Signal: Use of Float for financial data suggests a pattern of handling monetary values
# 🧠 ML Signal: Use of Integer for timestamps suggests a pattern of handling time-related data
# ✅ Best Practice: Use of Column with String type for order_type ensures proper database schema definition
# 🧠 ML Signal: register_schema function call indicates a pattern for setting up database schemas


#: the position for specific entity of every day
class Position(TraderBase, Mixin):
    __tablename__ = "position"

    #: 机器人名字
    trader_name = Column(String(length=128))
    #: 账户id
    account_stats_id = Column(Integer, ForeignKey("account_stats.id"))
    account_stats = relationship("AccountStats", back_populates="positions")

    #: 做多数量
    long_amount = Column(Float)
    #: 可平多数量
    available_long = Column(Float)
    #: 平均做多价格
    average_long_price = Column(Float)

    #: 做空数量
    short_amount = Column(Float)
    #: 可平空数量
    available_short = Column(Float)
    #: 平均做空价格
    average_short_price = Column(Float)

    #: 盈亏
    profit = Column(Float)
    #: 盈亏比例
    profit_rate = Column(Float)
    #: 市值 或者 占用的保证金(方便起见，总是100%)
    value = Column(Float)
    #: 交易类型(0代表T+0,1代表T+1)
    trading_t = Column(Integer)


#: 委托单
class Order(TraderBase, Mixin):
    __tablename__ = "order"

    #: 机器人名字
    trader_name = Column(String(length=128))
    #: 订单价格
    order_price = Column(Float)
    #: 订单数量
    order_amount = Column(Float)
    #: 订单类型
    order_type = Column(String(length=64))
    #: 订单状态
    status = Column(String(length=64))

    #: 产生订单的selector/factor level
    level = Column(String(length=32))


register_schema(providers=["zvt"], db_name="trader_info", schema_base=TraderBase)

# the __all__ is generated
__all__ = ["TraderInfo", "AccountStats", "Position", "Order"]
