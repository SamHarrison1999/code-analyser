# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy.ext.declarative import declarative_base

from zvt.contract.register import register_schema
# ✅ Best Practice: Naming convention for classes should follow CamelCase.
from zvt.factors.zen.domain.common import ZenFactorCommon

# 🧠 ML Signal: Usage of SQLAlchemy's declarative_base indicates ORM pattern.
# 🧠 ML Signal: Use of class inheritance and tablename for ORM mapping
Index1dZenFactorBase = declarative_base()
# ✅ Best Practice: Use of __all__ to define public API of the module
# 🧠 ML Signal: Registration of schema with specific providers and database name
# ✅ Best Practice: Explicitly specifying schema details for database registration


class Index1dZenFactor(Index1dZenFactorBase, ZenFactorCommon):
    __tablename__ = "index_1d_zen_factor"


register_schema(providers=["zvt"], db_name="index_1d_zen_factor", schema_base=Index1dZenFactorBase, entity_type="index")


# the __all__ is generated
__all__ = ["Index1dZenFactor"]