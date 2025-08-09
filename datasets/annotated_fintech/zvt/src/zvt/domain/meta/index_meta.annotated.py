# -*- coding: utf-8 -*-

# 🧠 ML Signal: Usage of declarative_base from SQLAlchemy indicates ORM pattern
from sqlalchemy import Column, String, Float
from sqlalchemy.orm import declarative_base
# 🧠 ML Signal: Importing custom modules indicates potential domain-specific logic

from zvt.contract import Portfolio, PortfolioStockHistory
# 🧠 ML Signal: Custom decorators suggest usage patterns for extending functionality
# ✅ Best Practice: Class inherits from multiple base classes, ensure method resolution order is as intended
from zvt.contract.register import register_schema, register_entity

# ✅ Best Practice: Using declarative_base for ORM base class is a standard practice
# 🧠 ML Signal: Use of class variable to define table name in ORM
IndexMetaBase = declarative_base()

# 🧠 ML Signal: Decorator usage indicates a pattern for registering entities
# 🧠 ML Signal: Use of Column with String type and length constraint

#: 指数
# 🧠 ML Signal: Use of Column with String type and length constraint
# 🧠 ML Signal: Inheritance from multiple classes, indicating a complex class hierarchy
@register_entity(entity_type="index")
class Index(IndexMetaBase, Portfolio):
    # 🧠 ML Signal: Use of __all__ to define public API of the module
    # 🧠 ML Signal: Use of Column with Float type
    # 🧠 ML Signal: Use of class variable for database table name, common in ORM patterns
    # 🧠 ML Signal: Function call with specific parameters, indicating schema registration pattern
    __tablename__ = "index"

    #: 发布商
    publisher = Column(String(length=64))
    #: 类别
    #: see IndexCategory
    category = Column(String(length=64))
    #: 基准点数
    base_point = Column(Float)


class IndexStock(IndexMetaBase, PortfolioStockHistory):
    __tablename__ = "index_stock"


register_schema(providers=["em", "exchange"], db_name="index_meta", schema_base=IndexMetaBase)


# the __all__ is generated
__all__ = ["Index", "IndexStock"]