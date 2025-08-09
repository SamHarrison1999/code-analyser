# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, Float

# ✅ Best Practice: Grouping related imports together improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin

# ✅ Best Practice: Naming convention for base classes should be consistent and descriptive.
from zvt.contract.register import register_schema

# ✅ Best Practice: Use of __tablename__ for ORM table naming

ValuationBase = declarative_base()
# ✅ Best Practice: Explicitly defining column types for ORM mapping


# ✅ Best Practice: Explicitly defining column types for ORM mapping
class StockValuation(ValuationBase, Mixin):
    __tablename__ = "stock_valuation"
    # ✅ Best Practice: Explicitly defining column types for ORM mapping

    code = Column(String(length=32))
    # ✅ Best Practice: Explicitly defining column types for ORM mapping
    name = Column(String(length=32))
    #: 总股本(股)
    # ✅ Best Practice: Explicitly defining column types for ORM mapping
    capitalization = Column(Float)
    #: 公司已发行的普通股股份总数(包含A股，B股和H股的总股本)
    # ✅ Best Practice: Explicitly defining column types for ORM mapping
    circulating_cap = Column(Float)
    #: 市值
    # 🧠 ML Signal: Inheritance from multiple classes indicates a pattern of using mixins for shared functionality.
    # ✅ Best Practice: Explicitly defining column types for ORM mapping
    market_cap = Column(Float)
    #: 流通市值
    # ✅ Best Practice: Explicitly defining column types for ORM mapping
    # 🧠 ML Signal: Use of SQLAlchemy ORM pattern for database table representation.
    circulating_market_cap = Column(Float)
    #: 换手率
    # 🧠 ML Signal: Consistent use of String type for identifiers.
    # ✅ Best Practice: Explicitly defining column types for ORM mapping
    turnover_ratio = Column(Float)
    #: 静态pe
    # 🧠 ML Signal: Use of Float type for financial metrics.
    # ✅ Best Practice: Explicitly defining column types for ORM mapping
    pe = Column(Float)
    #: 动态pe
    # ✅ Best Practice: Explicitly defining column types for ORM mapping
    pe_ttm = Column(Float)
    #: 市净率
    # ✅ Best Practice: Explicitly defining column types for ORM mapping
    pb = Column(Float)
    #: 市销率
    ps = Column(Float)
    #: 市现率
    # ✅ Best Practice: Use of __all__ to define public API of the module.
    # ✅ Best Practice: Consistent naming convention for financial metrics.
    # 🧠 ML Signal: Registration of schema with specific providers and database configuration.
    pcf = Column(Float)


class EtfValuation(ValuationBase, Mixin):
    __tablename__ = "etf_valuation"

    code = Column(String(length=32))
    name = Column(String(length=32))
    #: 静态pe
    pe = Column(Float)
    #: 加权
    pe1 = Column(Float)
    #: 动态pe
    pe_ttm = Column(Float)
    #: 加权
    pe_ttm1 = Column(Float)
    #: 市净率
    pb = Column(Float)
    #: 加权
    pb1 = Column(Float)
    #: 市销率
    ps = Column(Float)
    #: 加权
    ps1 = Column(Float)
    #: 市现率
    pcf = Column(Float)
    #: 加权
    pcf1 = Column(Float)


register_schema(
    providers=["joinquant"],
    db_name="valuation",
    schema_base=ValuationBase,
    entity_type="stock",
)


# the __all__ is generated
__all__ = ["StockValuation", "EtfValuation"]
