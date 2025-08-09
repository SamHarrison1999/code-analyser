# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Float
# ✅ Best Practice: Group related imports together for better readability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin
# ✅ Best Practice: Use a consistent naming convention for base classes.
from zvt.contract.register import register_schema

# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
EmotionBase = declarative_base()

# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction

class LimitUpInfo(EmotionBase, Mixin):
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    __tablename__ = "limit_up_info"

    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    code = Column(String(length=32))
    name = Column(String(length=32))
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 是否新股
    is_new = Column(Boolean)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 是否回封，是就是打开过，否相反
    is_again_limit = Column(Boolean)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 涨停打开次数,0代表封住就没开板
    open_count = Column(Integer)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 首次封板时间
    first_limit_up_time = Column(DateTime)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 最后封板时间
    # ✅ Best Practice: Use of __tablename__ for ORM table naming
    last_limit_up_time = Column(DateTime)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 涨停类型:换手板，一字板
    # 🧠 ML Signal: Use of Column with specific data types for ORM mapping
    limit_up_type = Column(String)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 封单金额
    # 🧠 ML Signal: Use of Column with specific data types for ORM mapping
    order_amount = Column(String)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 最近一年封板成功率
    # 🧠 ML Signal: Use of Boolean type for binary attributes
    success_rate = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 流通市值
    # 🧠 ML Signal: Use of Boolean type for binary attributes
    currency_value = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 涨幅
    # 🧠 ML Signal: Use of Float type for numerical attributes
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    change_pct = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 换手率
    # 🧠 ML Signal: Use of Float type for numerical attributes
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    turnover_rate = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    #: 涨停原因
    # 🧠 ML Signal: Use of Float type for numerical attributes
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    reason = Column(String)
    #: 几天几板
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    high_days = Column(String)
    #: 最近几板，不一定是连板
    # 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
    high_days_count = Column(Integer)
# ✅ Best Practice: Use of __all__ to define public API of the module
# 🧠 ML Signal: Usage of SQLAlchemy ORM for database modeling
# 🧠 ML Signal: Registration of schema with specific providers and database name


class LimitDownInfo(EmotionBase, Mixin):
    __tablename__ = "limit_down_info"

    code = Column(String(length=32))
    name = Column(String(length=32))
    #: 是否新股
    is_new = Column(Boolean)
    #: 是否回封，是就是打开过，否相反
    is_again_limit = Column(Boolean)
    #: 流通市值
    currency_value = Column(Float)
    #: 涨幅
    change_pct = Column(Float)
    #: 换手率
    turnover_rate = Column(Float)


class Emotion(EmotionBase, Mixin):
    __tablename__ = "emotion"
    #: 涨停数量
    limit_up_count = Column(Integer)
    #: 炸板数
    limit_up_open_count = Column(Integer)
    #: 涨停封板成功率
    limit_up_success_rate = Column(Float)

    #: 连板高度
    max_height = Column(Integer)
    #: 连板数x个数 相加
    continuous_power = Column(Integer)

    #: 跌停数量
    limit_down_count = Column(Integer)
    #: 跌停打开
    limit_down_open_count = Column(Integer)
    #: 跌停封板成功率
    limit_down_success_rate = Column(Float)


register_schema(providers=["jqka"], db_name="emotion", schema_base=EmotionBase)


# the __all__ is generated
__all__ = ["LimitUpInfo", "LimitDownInfo", "Emotion"]