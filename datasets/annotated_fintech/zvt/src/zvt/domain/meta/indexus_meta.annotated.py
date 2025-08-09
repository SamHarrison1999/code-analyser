# -*- coding: utf-8 -*-

# ✅ Best Practice: Grouping related imports together improves readability.
from sqlalchemy import Column, String, Float
from sqlalchemy.orm import declarative_base

from zvt.contract import Portfolio

# ✅ Best Practice: Naming convention for base classes should be clear and descriptive.
from zvt.contract.register import register_schema, register_entity

# 🧠 ML Signal: Inheritance from multiple classes, indicating a potential pattern for class design

IndexusMetaBase = declarative_base()
# 🧠 ML Signal: Use of decorators to register entities can be a pattern for ML models to learn.
# 🧠 ML Signal: Use of SQLAlchemy ORM for database table mapping


# 🧠 ML Signal: Definition of a string column with a specific length
#: 美股指数
# 🧠 ML Signal: Definition of a string column with a specific length
# 🧠 ML Signal: Registration of schema with specific providers and database name
# 🧠 ML Signal: Definition of a float column, indicating numerical data storage
# ✅ Best Practice: Use of __all__ to define public interface of the module
@register_entity(entity_type="indexus")
class Indexus(IndexusMetaBase, Portfolio):
    __tablename__ = "index"

    #: 发布商
    publisher = Column(String(length=64))
    #: 类别
    #: see IndexCategory
    category = Column(String(length=64))
    #: 基准点数
    base_point = Column(Float)


register_schema(providers=["em"], db_name="indexus_meta", schema_base=IndexusMetaBase)


# the __all__ is generated
__all__ = ["Indexus"]
