# -*- coding: utf-8 -*-
# 🧠 ML Signal: Usage of SQLAlchemy's declarative_base to define ORM base class

from sqlalchemy.orm import declarative_base
# 🧠 ML Signal: Importing custom decorators for schema and entity registration

from zvt.contract.register import register_schema, register_entity
# 🧠 ML Signal: Importing a base class for tradable entities, indicating a financial domain
from zvt.contract.schema import TradableEntity

# 🧠 ML Signal: Use of class inheritance, indicating a design pattern for extending functionality
CurrencyMetaBase = declarative_base()
# 🧠 ML Signal: Function call with specific parameters, indicating a pattern for schema registration
# ⚠️ SAST Risk (Low): Potential risk if `register_schema` does not validate inputs properly
# ✅ Best Practice: Use of `__all__` to define public symbols of the module
# 🧠 ML Signal: Defining a new SQLAlchemy ORM base class for currency-related entities
# 🧠 ML Signal: Usage of a custom decorator to register an entity type, indicating extensibility


@register_entity(entity_type="currency")
class Currency(CurrencyMetaBase, TradableEntity):
    __tablename__ = "currency"


register_schema(providers=["em"], db_name="currency_meta", schema_base=CurrencyMetaBase)


# the __all__ is generated
__all__ = ["Currency"]