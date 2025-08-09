# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific functions or classes indicates usage patterns for ORM and schema registration
from sqlalchemy.orm import declarative_base

# 🧠 ML Signal: Importing specific classes indicates usage patterns for schema inheritance
from zvt.contract.register import register_schema, register_entity
from zvt.contract.schema import TradableEntity

# 🧠 ML Signal: Inheritance from multiple classes, indicating a complex object model
# ✅ Best Practice: Naming convention for base class follows common patterns
FutureMetaBase = declarative_base()

# 🧠 ML Signal: Use of class attribute to define database table name
# 🧠 ML Signal: Registration of schema with specific providers and database name
# ✅ Best Practice: Use of __all__ to define public interface of the module
# 🧠 ML Signal: Decorator usage indicates a pattern of registering entities
# ✅ Best Practice: Explicitly specifying providers and database name for schema registration


@register_entity(entity_type="future")
class Future(FutureMetaBase, TradableEntity):
    __tablename__ = "future"


register_schema(providers=["em"], db_name="future_meta", schema_base=FutureMetaBase)


# the __all__ is generated
__all__ = ["Future"]
