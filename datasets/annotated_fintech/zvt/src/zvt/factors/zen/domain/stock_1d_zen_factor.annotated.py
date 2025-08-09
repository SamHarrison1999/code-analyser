# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy.ext.declarative import declarative_base

from zvt.contract.register import register_schema
# 🧠 ML Signal: Usage of SQLAlchemy's declarative_base indicates ORM pattern.
from zvt.factors.zen.domain.common import ZenFactorCommon
# 🧠 ML Signal: Inheritance from multiple base classes, indicating a pattern of using mixins or shared functionality.

Stock1dZenFactorBase = declarative_base()
# 🧠 ML Signal: Use of class-level attributes to define database table names, common in ORM patterns.
# 🧠 ML Signal: Registration of schema with specific providers and database names, indicating a pattern of dynamic schema management.
# ✅ Best Practice: Explicitly listing all public symbols in __all__ to define the module's public API.


class Stock1dZenFactor(Stock1dZenFactorBase, ZenFactorCommon):
    __tablename__ = "stock_1d_zen_factor"


register_schema(providers=["zvt"], db_name="stock_1d_zen_factor", schema_base=Stock1dZenFactorBase, entity_type="stock")


# the __all__ is generated
__all__ = ["Stock1dZenFactor"]