# -*- coding: utf-8 -*-
# 🧠 ML Signal: Usage of SQLAlchemy for ORM, indicating a pattern of database interaction

from sqlalchemy.orm import declarative_base

# 🧠 ML Signal: Importing a specific class from a module, indicating a pattern of modular code usage

from zvt.contract import TradableEntity

# 🧠 ML Signal: Importing specific functions for registration, indicating a pattern of dynamic schema/entity registration
from zvt.contract.register import register_schema, register_entity

# ✅ Best Practice: Use of class inheritance to promote code reuse and organization

# 🧠 ML Signal: Creation of a base class for ORM models, indicating a pattern of database schema definition
CBondBase = declarative_base()
# ✅ Best Practice: Naming convention for base class follows common Python and SQLAlchemy practices
# 🧠 ML Signal: Usage of a decorator for entity registration, indicating a pattern of metadata or behavior extension
# 🧠 ML Signal: Use of class-level attributes to define database table names
# 🧠 ML Signal: Registration of schema with specific providers and database name
# ✅ Best Practice: Use of __all__ to define public interface of the module


#: 美股
@register_entity(entity_type="cbond")
class CBond(CBondBase, TradableEntity):
    __tablename__ = "cbond"


register_schema(providers=["em"], db_name="cbond_meta", schema_base=CBondBase)


# the __all__ is generated
__all__ = ["CBond"]
