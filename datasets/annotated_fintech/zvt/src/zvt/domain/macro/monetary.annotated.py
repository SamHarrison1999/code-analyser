# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, Float
# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin
from zvt.contract.register import register_schema
# ✅ Best Practice: Naming convention for SQLAlchemy base class is clear and descriptive.
# 🧠 ML Signal: Inheritance from MonetaryBase and Mixin indicates a pattern of using mixins for shared functionality.

MonetaryBase = declarative_base()
# 🧠 ML Signal: Use of SQLAlchemy's Column and String/Float types indicates a pattern of ORM usage.


# 🧠 ML Signal: Use of SQLAlchemy's Column and String/Float types indicates a pattern of ORM usage.
class TreasuryYield(MonetaryBase, Mixin):
    __tablename__ = "treasury_yield"
    # 🧠 ML Signal: Use of SQLAlchemy's Column and String/Float types indicates a pattern of ORM usage.

    # ✅ Best Practice: Use of __all__ to define public API of the module.
    # 🧠 ML Signal: Use of SQLAlchemy's Column and String/Float types indicates a pattern of ORM usage.
    # ⚠️ SAST Risk (Low): Ensure that the register_schema function handles inputs securely to prevent injection attacks.
    code = Column(String(length=32))

    # 2年期
    yield_2 = Column(Float)
    # 5年期
    yield_5 = Column(Float)
    # 10年期
    yield_10 = Column(Float)
    # 30年期
    yield_30 = Column(Float)


register_schema(providers=["em"], db_name="monetary", schema_base=MonetaryBase)


# the __all__ is generated
__all__ = ["TreasuryYield"]