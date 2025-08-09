# -*- coding: utf-8 -*-

# 🧠 ML Signal: Usage of custom base class for ORM models
from sqlalchemy import Column, String, DateTime, BigInteger, Float
from sqlalchemy.orm import declarative_base
# 🧠 ML Signal: Usage of custom schema registration

from zvt.contract import TradableEntity
from zvt.contract.register import register_schema, register_entity
# 🧠 ML Signal: Custom base class for SQLAlchemy models
# ✅ Best Practice: Use of class-level variable for table name improves maintainability

StockMetaBase = declarative_base()
# ✅ Best Practice: Use of DateTime for date fields is appropriate for storing timestamps

# 🧠 ML Signal: Decorator usage for entity registration

# ✅ Best Practice: Use of String for text fields is appropriate for storing string data
#: 个股
@register_entity(entity_type="stock")
# ✅ Best Practice: Use of String for text fields is appropriate for storing string data
class Stock(StockMetaBase, TradableEntity):
    __tablename__ = "stock"
    # ✅ Best Practice: Use of Float for numeric fields is appropriate for storing decimal numbers
    #: 股东上次更新时间
    holder_modified_date = Column(DateTime)
    #: 控股股东
    controlling_holder = Column(String)
    #: 实际控制人
    controlling_holder_parent = Column(String)
    #: 前十大股东占比
    top_ten_ratio = Column(Float)


#: 个股详情
# 🧠 ML Signal: Usage of register_schema function indicates schema registration pattern
class StockDetail(StockMetaBase, TradableEntity):
    __tablename__ = "stock_detail"

    # 🧠 ML Signal: __all__ usage indicates module export pattern
    #: 所属行业
    industries = Column(String)
    #: 行业指数
    industry_indices = Column(String)
    #: 所属板块
    concept_indices = Column(String)
    #: 所属区域
    area_indices = Column(String)

    #: 成立日期
    date_of_establishment = Column(DateTime)
    #: 公司简介
    profile = Column(String(length=1024))
    #: 主营业务
    main_business = Column(String(length=512))
    #: 发行量(股)
    issues = Column(BigInteger)
    #: 发行价格
    price = Column(Float)
    #: 募资净额(元)
    raising_fund = Column(Float)
    #: 发行市盈率
    issue_pe = Column(Float)
    #: 网上中签率
    net_winning_rate = Column(Float)


register_schema(
    providers=["exchange", "joinquant", "eastmoney", "em", "qmt"], db_name="stock_meta", schema_base=StockMetaBase
)


# the __all__ is generated
__all__ = ["Stock", "StockDetail"]