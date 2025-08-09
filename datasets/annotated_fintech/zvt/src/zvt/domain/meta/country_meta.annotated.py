# -*- coding: utf-8 -*-

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy import Column, String, Float
from sqlalchemy.orm import declarative_base

from zvt.contract.register import register_schema, register_entity

# ✅ Best Practice: Naming convention for base classes should be clear and descriptive.
from zvt.contract.schema import TradableEntity

# 🧠 ML Signal: Inheritance from multiple classes indicates a complex object model

CountryMetaBase = declarative_base()
# 🧠 ML Signal: Use of decorators to register entities indicates a pattern for extensibility and modularity.
# 🧠 ML Signal: Database table name definition for ORM


# 🧠 ML Signal: Column definition with data type and length
@register_entity(entity_type="country")
class Country(CountryMetaBase, TradableEntity):
    # 🧠 ML Signal: Column definition with data type and length
    __tablename__ = "country"

    # 🧠 ML Signal: Schema registration with specific providers and database name
    # 🧠 ML Signal: Column definition with data type and length
    # ✅ Best Practice: Use of __all__ to define public API of the module
    #: 区域
    #: region
    region = Column(String(length=128))
    #: 首都
    #: capital city
    capital_city = Column(String(length=128))
    #: 收入水平
    #: income level
    income_level = Column(String(length=64))
    #: 贷款类型
    #: lending type
    lending_type = Column(String(length=64))
    #: 经度
    #: longitude
    longitude = Column(Float)
    #: 纬度
    #: latitude
    latitude = Column(Float)


register_schema(providers=["wb"], db_name="country_meta", schema_base=CountryMetaBase)


# the __all__ is generated
__all__ = ["Country"]
