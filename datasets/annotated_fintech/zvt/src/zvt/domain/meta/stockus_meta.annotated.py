# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing a base class for ORM models, indicating use of SQLAlchemy for database interactions

from sqlalchemy.orm import declarative_base
# 🧠 ML Signal: Importing a specific class from a module, indicating a dependency on the zvt library

from zvt.contract import TradableEntity
# 🧠 ML Signal: Importing functions for registration, indicating dynamic schema/entity registration
from zvt.contract.register import register_schema, register_entity
# 🧠 ML Signal: Inheritance from multiple classes indicates a design pattern that could be learned.

# 🧠 ML Signal: Creating a base class for ORM models, indicating a pattern of using SQLAlchemy for ORM
StockusMetaBase = declarative_base()
# ✅ Best Practice: Use of __all__ to define public API of the module.
# ✅ Best Practice: Naming convention for base classes in SQLAlchemy is clear and descriptive
# 🧠 ML Signal: Using a decorator for entity registration, indicating a pattern of dynamic entity management
# ✅ Best Practice: Use of decorators for registration improves code readability and organization
# 🧠 ML Signal: Use of class attribute for table name suggests ORM usage pattern.
# 🧠 ML Signal: Function call with specific parameters indicates a pattern for schema registration.
# ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the register_schema call.


#: 美股
@register_entity(entity_type="stockus")
class Stockus(StockusMetaBase, TradableEntity):
    __tablename__ = "stockus"


register_schema(providers=["em"], db_name="stockus_meta", schema_base=StockusMetaBase)


# the __all__ is generated
__all__ = ["Stockus"]