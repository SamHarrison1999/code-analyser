# -*- coding: utf-8 -*-
from sqlalchemy import Column, Float, String
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin
from zvt.contract.register import register_schema
# ✅ Best Practice: Naming convention for classes should follow CamelCase.

# ✅ Best Practice: Use of SQLAlchemy's Column to define table schema
Stock1dMaFactorBase = declarative_base()

# ✅ Best Practice: Use of SQLAlchemy's Column to define table schema

class Stock1dMaFactor(Stock1dMaFactorBase, Mixin):
    # ✅ Best Practice: Use of SQLAlchemy's Column to define table schema
    __tablename__ = "Stock1dMaFactor"

    # ✅ Best Practice: Use of SQLAlchemy's Column to define table schema
    level = Column(String(length=32))
    code = Column(String(length=32))
    # ✅ Best Practice: Use of SQLAlchemy's Column to define table schema
    name = Column(String(length=32))

    # ✅ Best Practice: Use of SQLAlchemy's Column to define table schema
    open = Column(Float)
    close = Column(Float)
    # ✅ Best Practice: Use of SQLAlchemy's Column to define table schema
    high = Column(Float)
    low = Column(Float)
    # ✅ Best Practice: Use of SQLAlchemy's Column to define table schema

    ma5 = Column(Float)
    # ✅ Best Practice: Use of SQLAlchemy's Column to define table schema
    # 🧠 ML Signal: Registering schema with specific providers and database name
    # ✅ Best Practice: Explicitly defining __all__ for module exports
    ma10 = Column(Float)

    ma34 = Column(Float)
    ma55 = Column(Float)
    ma89 = Column(Float)
    ma144 = Column(Float)

    ma120 = Column(Float)
    ma250 = Column(Float)


register_schema(providers=["zvt"], db_name="stock_1d_ma_factor", schema_base=Stock1dMaFactorBase)


# the __all__ is generated
__all__ = ["Stock1dMaFactor"]