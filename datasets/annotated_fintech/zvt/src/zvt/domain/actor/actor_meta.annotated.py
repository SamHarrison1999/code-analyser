# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.

from sqlalchemy.orm import declarative_base

# 🧠 ML Signal: Usage of SQLAlchemy's declarative_base indicates ORM pattern.
from zvt.contract.register import register_schema

# ✅ Best Practice: Define __tablename__ for ORM mapping clarity
from zvt.contract.schema import ActorEntity

# ✅ Best Practice: Use of __all__ to define public API of the module
# 🧠 ML Signal: Usage of a custom function to register schema
ActorMetaBase = declarative_base()


#: 参与者
class ActorMeta(ActorMetaBase, ActorEntity):
    __tablename__ = "actor_meta"


register_schema(providers=["em"], db_name="actor_meta", schema_base=ActorMetaBase)


# the __all__ is generated
__all__ = ["ActorMeta"]
