# -*- coding: utf-8 -*-

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy import Column, String
from sqlalchemy.orm import declarative_base
# ✅ Best Practice: Grouping imports from the same module together improves readability.

from zvt.contract import Portfolio, PortfolioStock
from zvt.contract.register import register_schema, register_entity
# 🧠 ML Signal: Inheritance from multiple classes, indicating a complex class hierarchy
# ✅ Best Practice: Naming convention for base classes should be clear and descriptive.

BlockMetaBase = declarative_base()
# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction

# 🧠 ML Signal: Inheritance from multiple classes, indicating a mix of behaviors and properties
# 🧠 ML Signal: Use of decorators to register entities indicates a pattern for extensibility and modularity.

# 🧠 ML Signal: Definition of a database column with a specific string length
#: 板块
# 🧠 ML Signal: Use of class attribute to define table name, common in ORM patterns
# ✅ Best Practice: Use of __all__ to define public API of the module
# 🧠 ML Signal: Function call with specific parameters, indicating configuration or setup pattern
@register_entity(entity_type="block")
class Block(BlockMetaBase, Portfolio):
    __tablename__ = "block"

    #: 板块类型，行业(industry),概念(concept)
    category = Column(String(length=64))


class BlockStock(BlockMetaBase, PortfolioStock):
    __tablename__ = "block_stock"


register_schema(providers=["em", "eastmoney", "sina"], db_name="block_meta", schema_base=BlockMetaBase)


# the __all__ is generated
__all__ = ["Block", "BlockStock"]