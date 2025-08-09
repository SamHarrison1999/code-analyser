# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, Float, BIGINT
# ✅ Best Practice: Group related imports together for better readability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin
# ✅ Best Practice: Use a consistent naming convention for base classes.
from zvt.contract.register import register_schema

# 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
MacroBase = declarative_base()

# 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling

class Economy(MacroBase, Mixin):
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    # https://datatopics.worldbank.org/world-development-indicators//themes/economy.html
    __tablename__ = "economy"
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling

    code = Column(String(length=32))
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    name = Column(String(length=32))
    population = Column(BIGINT)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling

    gdp = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    gdp_per_capita = Column(Float)
    gdp_per_employed = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    gdp_growth = Column(Float)
    agriculture_growth = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    industry_growth = Column(Float)
    manufacturing_growth = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    service_growth = Column(Float)
    consumption_growth = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    capital_growth = Column(Float)
    exports_growth = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
    # 🧠 ML Signal: Registration of schema with specific providers and database name
    # ✅ Best Practice: Use of __all__ to define public API of the module
    imports_growth = Column(Float)

    gni = Column(Float)
    gni_per_capita = Column(Float)

    gross_saving = Column(Float)
    cpi = Column(Float)
    unemployment_rate = Column(Float)
    fdi_of_gdp = Column(Float)


register_schema(providers=["wb"], db_name="macro", schema_base=MacroBase)


# the __all__ is generated
__all__ = ["Economy"]